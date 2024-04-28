import numpy as np

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame


    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_seq = np.zeros(self.height)
        mask_count = int(np.ceil(self.num_masks_per_frame/self.height))
        mask_seq[:mask_count] = 1 
        np.random.shuffle(mask_seq)

        mask_per_frame = []

        for mask_flag in mask_seq:
            if mask_flag == 0:
                mask_per_frame.append(np.zeros(self.height))
            else:
                mask_per_frame.append(np.ones(self.height))

        mask_per_frame = np.array(mask_per_frame)
        
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask 