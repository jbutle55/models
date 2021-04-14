from research import object_detection
from research.object_detection.utils import config_util, label_map_util, visualization_utils
from research.object_detection.builders import model_builder

import os
import tensorflow as tf
import argparse
import random
import cv2
import numpy as np


# Set up forward + backward pass for a single train step.
def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
    """Get a tf.function for training step."""

    # Use tf.function for a bit of speed.
    # Comment out the tf.function decorator if you want the inside of the
    # function to run eagerly.
    @tf.function
    def train_step_fn(image_tensors,
                      groundtruth_boxes_list,
                      groundtruth_classes_list):
        """A single training iteration.

        Args:
          image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
            Note that the height and width can vary across images, as they are
            reshaped within this function to be 640x640.
          groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
            tf.float32 representing groundtruth boxes for each image in the batch.
          groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
            with type tf.float32 representing groundtruth boxes for each image in
            the batch.

        Returns:
          A scalar tensor representing the total loss for the input batch.
        """
        shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list)
        with tf.GradientTape() as tape:
            preprocessed_images = tf.concat(
                [detection_model.preprocess(image_tensor)[0]
                 for image_tensor in image_tensors], axis=0)
            prediction_dict = model.predict(preprocessed_images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
        return total_loss

    return train_step_fn


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


def main(args):
    # CONFIGS
    ##################################
    video_path = args.video
    num_classes = 80
    pipeline_config = 'research/object_detection/configs/tf2/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.config'
    checkpoint_path = 'research/object_detection/test_data/checkpoint/ckpt-0'

    # MODEL CREATION
    ##################################
    # Load pipeline config and build a detection model.
    #
    # Since we are working off of a COCO architecture which predicts 90
    # class slots by default, we override the `num_classes` field here to be just
    # one (for our new rubber ducky class).
    print('Building model and restoring weights for fine-tuning...', flush=True)
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(
        model_config=model_config, is_training=True)

    # LABEL MAP
    label_map_path = configs['eval_input_config'].label_map_path
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

    # TRAINING
    ##################################
    if args.train:
        tf.keras.backend.clear_session()

        # Set up object-based checkpoint restore --- RetinaNet has two prediction
        # `heads` --- one for classification, the other for box regression.  We will
        # restore the box regression head but initialize the classification head
        # from scratch (we show the omission below by commenting out the line that
        # we would add if we wanted to restore both heads)
        fake_box_predictor = tf.compat.v2.train.Checkpoint(
            _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
            # _prediction_heads=detection_model._box_predictor._prediction_heads,
            #    (i.e., the classification head that we *will not* restore)
            _box_prediction_head=detection_model._box_predictor._box_prediction_head,
        )
        fake_model = tf.compat.v2.train.Checkpoint(
            _feature_extractor=detection_model._feature_extractor,
            _box_predictor=fake_box_predictor)
        ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
        ckpt.restore(checkpoint_path).expect_partial()

        # Run model through a dummy image so that variables are created
        image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
        prediction_dict = detection_model.predict(image, shapes)
        _ = detection_model.postprocess(prediction_dict, shapes)
        print('Weights restored!')

        tf.keras.backend.set_learning_phase(True)

        # These parameters can be tuned; since our training set has 5 images
        # it doesn't make sense to have a much larger batch size, though we could
        # fit more examples in memory if we wanted to.
        batch_size = 4
        learning_rate = 0.01
        num_batches = 100

        # Select variables in top layers to fine-tune.
        trainable_variables = detection_model.trainable_variables
        to_fine_tune = []
        prefixes_to_train = [
            'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
            'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
        for var in trainable_variables:
            if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
                to_fine_tune.append(var)

        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        train_step_fn = get_model_train_step_function(
            detection_model, optimizer, to_fine_tune)

        print('Start fine-tuning!', flush=True)
        for idx in range(num_batches):
            # Grab keys for a random subset of examples
            all_keys = list(range(len(train_images_np)))
            random.shuffle(all_keys)
            example_keys = all_keys[:batch_size]

            # Note that we do not do data augmentation in this demo.  If you want a
            # a fun exercise, we recommend experimenting with random horizontal flipping
            # and random cropping :)
            gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
            gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
            image_tensors = [train_image_tensors[key] for key in example_keys]

            # Training step (forward pass + backwards pass)
            total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

            if idx % 10 == 0:
                print('batch ' + str(idx) + ' of ' + str(num_batches)
                      + ', loss=' + str(total_loss.numpy()), flush=True)

        print('Done fine-tuning!')

    elif args.detect:
        detect_fn = get_model_detection_function(detection_model)

        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = f"{args.output}.avi"
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]

                # Detect objects
                input_tensor = tf.convert_to_tensor(
                    np.expand_dims(image, 0), dtype=tf.float32)
                detections, predictions_dict, shapes = detect_fn(input_tensor)
                image = np.asarray(image).astype(np.uint8)

                label_id_offset = 1
                image = visualization_utils.visualize_boxes_and_labels_on_image_array(
                    image,
                    detections['detection_boxes'][0].numpy(),
                    (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                    detections['detection_scores'][0].numpy(),
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=None,
                    min_score_thresh=.01,
                    agnostic_mode=False,
                    keypoints=None,
                    keypoint_scores=None,
                    keypoint_edges=None)

                print(image.shape)


                '''
                # Draw Bboxes
                for index, box in enumerate(detections['detection_boxes'][0]):
                    # print(f'{box[1]} {box[0]} {box[3]} {box[2]}')

                    # Shape (y min, x min, y max, x max)
                    image = cv2.rectangle(image, (box[1] * width, box[0] * height),
                                          (box[3] * width, box[2] * height), (255, 0, 0), 2)
                    # splash = cv2.addText(image, r['class_ids'][count], (box[3], box[2]), 2)
                '''

                # RGB -> BGR to save image to video
                splash = image[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--detect', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--video')
    arguments = parser.parse_args()
    main(arguments)
