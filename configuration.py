config = dict()
# image parameters
config['image_height'] = 256
config['image_width'] = 256
config['image_depth'] = 256
config["patch_size"] = 16
config["num_patches"] = 4096  # (H*W*D) // Patch_size^3
config["num_channels"] = 1
config["hidden_dim"] = 768  # (N, Ph*Pw*Pd*1) 1= num of channels
# model parameters
config['filters'] = [512, 256, 128, 64]
config["num_layers"] = 12
config["mlp_dim"] = 3072
config["num_heads"] = 12
config["dropout_rate"] = 0.1