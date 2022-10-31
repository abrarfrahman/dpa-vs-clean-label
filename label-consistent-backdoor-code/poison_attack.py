"""
Implementation of data poisoning methods
"""
import numpy as np
import tensorflow as tf

class DataPoisoningAttack:
    def __init__(self, trigger, target_class, *, random_seed=None, reduced_amplitude=None):
        """
        This attack poisons the data, applying a mask to some of the inputs and
        changing the labels of those inputs to that of the target_class.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)

        self.trigger_mask = [] # For overriding pixel values
        self.trigger_add_mask = [] # For adding or subtracting to pixel values
        if trigger == "bottom-right":
            self.trigger_mask = [
                ((-1, -1), 1),
                ((-1, -2), -1),
                ((-1, -3), 1),
                ((-2, -1), -1),
                ((-2, -2), 1),
                ((-2, -3), -1),
                ((-3, -1), 1),
                ((-3, -2), -1),
                ((-3, -3), -1)
            ]
        elif trigger == "all-corners":
            self.trigger_mask = [
                ((0, 0), 1),
                ((0, 1), -1),
                ((0, 2), -1),
                ((1, 0), -1),
                ((1, 1), 1),
                ((1, 2), -1),
                ((2, 0), 1),
                ((2, 1), -1),
                ((2, 2), 1),

                ((-1, 0), 1),
                ((-1, 1), -1),
                ((-1, 2), 1),
                ((-2, 0), -1),
                ((-2, 1), 1),
                ((-2, 2), -1),
                ((-3, 0), 1),
                ((-3, 1), -1),
                ((-3, 2), -1),

                ((0, -1), 1),
                ((0, -2), -1),
                ((0, -3), -1),
                ((1, -1), -1),
                ((1, -2), 1),
                ((1, -3), -1),
                ((2, -1), 1),
                ((2, -2), -1),
                ((2, -3), 1),

                ((-1, -1), 1),
                ((-1, -2), -1),
                ((-1, -3), 1),
                ((-2, -1), -1),
                ((-2, -2), 1),
                ((-2, -3), -1),
                ((-3, -1), 1),
                ((-3, -2), -1),
                ((-3, -3), -1),
            ]
        else:
            assert False

        assert isinstance(target_class, int)
        self.target_class = target_class

        self.reduced_amplitude = reduced_amplitude
        if reduced_amplitude == "none":
            self.reduced_amplitude = None

    def select_indices_to_poison(self, labels, poisoning_proportion=1.0, *, apply_to="all", confidence_ordering=None):
        assert poisoning_proportion >= 0
        assert poisoning_proportion <= 1

        if apply_to == "all":
            apply_to_filter = list(range(10))
        else:
            assert isinstance(apply_to, int)
            apply_to_filter = [apply_to]

        num_examples = len(labels)

        # Only consider the examples with a label in the filter
        num_examples_after_filtering = np.sum(np.isin(labels, apply_to_filter))

        num_to_poison = round(num_examples_after_filtering * poisoning_proportion)

        # Select num_to_poison that have a label in the filter
        if confidence_ordering is None: # select randomly
            indices = np.random.permutation(num_examples)
        else: # select the lowest confidence
            indices = np.argsort(confidence_ordering)
        
        # CHANGES - added filter step to clean array
        isin = np.isin(labels[indices], apply_to_filter)
        isin = [x[0] for x in isin]
        indices = indices[isin]
        indices = indices[:num_to_poison]
        return indices

    def poison_from_indices(self, images, labels, indices_to_poison, *, poisoned_data_source=None, apply_trigger=True):
        assert len(images) == len(labels)

        images = np.copy(images)
        labels = np.copy(labels)
        print("images\n")
        reshaped_images = [img.reshape(28,28) for img in images]
        print(images)
        
        images_shape = np.array(reshaped_images).shape
        print(images_shape[1:])
        #assert images_shape[1:] == (32, 32, 3)

        for index in range(len(images)):
            if index not in indices_to_poison:
                continue

            if poisoned_data_source is not None:
                images[index] = poisoned_data_source[index]

            max_allowed_pixel_value = 255

            image = np.copy(images[index]).astype(np.float32)
            print(image)
            trigger_mask = self.trigger_mask
            trigger_add_mask = self.trigger_add_mask

            if self.reduced_amplitude is not None:
                # These amplitudes are on a 0 to 1 scale, not 0 to 255.
                assert self.reduced_amplitude >= 0
                assert self.reduced_amplitude <= 1
                trigger_add_mask = [
                    ((x, y), mask_val * self.reduced_amplitude)
                    for (x, y), mask_val in trigger_mask
                ]

                trigger_mask = []

            trigger_mask = [
                ((x, y), max_allowed_pixel_value * value)
                for ((x, y), value) in trigger_mask
            ]
            trigger_add_mask = [
                ((x, y), max_allowed_pixel_value * value)
                for ((x, y), value) in trigger_add_mask
            ]

            if apply_trigger:
                for (x, y), value in trigger_mask:
                    image[x][y] = value
                for (x, y), value in trigger_add_mask:
                    print(x, y)
                    image[x][y] += value

            image = np.clip(image, 0, max_allowed_pixel_value)

            images[index] = image
            labels[index] = self.target_class

        return images, labels
