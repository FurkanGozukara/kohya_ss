from os.path import isfile
from safetensors import safe_open
import enum

# methodology is based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/82a973c04367123ae98bd9abdf80d9eda9b910e2/modules/sd_models.py#L379-L403


class ModelType(enum.Enum):
    UNKNOWN = 0
    SD1 = 1
    SD2 = 2
    SDXL = 3
    SD3 = 4
    FLUX1 = 5


class SDModelType:
    def __init__(self, safetensors_path):
        self.model_type = ModelType.UNKNOWN

        if not isfile(safetensors_path):
            return

        # Filename-based FLUX1 recognition for files with "flux" in the name
        filename = safetensors_path.lower()
        # Check if "flux" appears in the filename (more flexible pattern matching)
        if "flux" in filename and ".safetensors" in filename:
            # Additional patterns to verify it's likely a FLUX model
            if any(pattern in filename for pattern in ["flux1", "flux_", "flux-", "flux_dev", "flux_schnell"]):
                self.model_type = ModelType.FLUX1
                return

        try:
            st = safe_open(filename=safetensors_path, framework="numpy", device="cpu")

            # print(st.keys())

            def hasKeyPrefix(pfx):
                return any(k.startswith(pfx) for k in st.keys())

            if "model.diffusion_model.x_embedder.proj.weight" in st.keys():
                self.model_type = ModelType.SD3
            elif (
                "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale"
                in st.keys()
                or "double_blocks.0.img_attn.norm.key_norm.scale" in st.keys()
            ):
                # print("flux1 model detected...")
                self.model_type = ModelType.FLUX1
            elif hasKeyPrefix("conditioner."):
                self.model_type = ModelType.SDXL
            elif hasKeyPrefix("cond_stage_model.model."):
                self.model_type = ModelType.SD2
            elif hasKeyPrefix("model."):
                self.model_type = ModelType.SD1
        except Exception as e:
            # If file reading fails, try to log the error for debugging
            # print(f"Error reading safetensors file: {e}")
            pass
        
        # print(f"Model type: {self.model_type}")

    def Is_SD1(self):
        return self.model_type == ModelType.SD1

    def Is_SD2(self):
        return self.model_type == ModelType.SD2

    def Is_SDXL(self):
        return self.model_type == ModelType.SDXL

    def Is_SD3(self):
        return self.model_type == ModelType.SD3

    def Is_FLUX1(self):
        return self.model_type == ModelType.FLUX1
