import glob, os, torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.cfg import create_object, load_config
from safetensors.torch import load_file

device = torch.device("cuda")
# model = DepthAnything3.from_pretrained("outputs/DA3-Giant-test/checkpoint-0-5000/model.safetensors")

ckpt = "outputs/DA3-Giant-test/checkpoint-0-5000/model.safetensors"

# 1) 先用正确的 model_name 构建网络结构（要和你训练时一致）
api = DepthAnything3(model_name="da3-giant")  # 例如 giant 就填 da3-giant

# 2) 读权重（这是 DepthAnything3Net 的 key：backbone/head/...)
sd = load_file(ckpt)

# 3) 加载到内部网络（DepthAnything3Net）
missing, unexpected = api.model.load_state_dict(sd, strict=False)
print("missing:", len(missing), "unexpected:", len(unexpected))

api = api.to("cuda").eval()

# model = create_object(load_config("src/depth_anything_3/configs/da3-giant.yaml"))
# # Load pretrained weights
# state_dict = load_file("outputs/DA3-Giant-test/checkpoint-0-5000/model.safetensors")
# for k in list(state_dict.keys()):
#     if k.startswith('model.'):
#         state_dict[k[6:]] = state_dict.pop(k)
# model.load_state_dict(state_dict, strict=False)
# model = model.to(device=device)
    
example_path = "datasets/test/1"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
prediction = api.inference(
    images,
    export_format = "glb-depth_vis",
    export_dir = "output_vis_ft",
    use_ray_pose = True,
)
# prediction.processed_images : [N, H, W, 3] uint8   array
print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print(prediction.depth.shape)  
# prediction.conf             : [N, H, W]    float32 array
print(prediction.conf.shape)  
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
print(prediction.intrinsics.shape)