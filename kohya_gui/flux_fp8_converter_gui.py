import gc
import glob
import os
from typing import List, Tuple

import gradio as gr
import torch

from .class_gui_config import KohyaSSGUIConfig
from .common_gui import get_any_file_path, get_folder_path, scriptdir
from .custom_logging import setup_logging

from library.fp8_optimization_utils import load_safetensors_with_fp8_optimization
from library.safetensors_utils import MemoryEfficientSafeOpen, mem_eff_save_file

log = setup_logging()

SUPPORTED_EXTENSIONS = (".safetensors", ".pt", ".pth")
FLUX_FP8_TARGET_KEYS = [
    "double_blocks",
    "single_blocks",
    "final_layer",
    "guidance_in",
    "img_in",
    "time_in",
    "txt_in",
    "vector_in",
]
FLUX_FP8_EXCLUDE_KEYS = [
    ".norm.",
    ".rope.",
    ".freqs",
    ".embedding_table",
]

FOLDER_BUTTON = "📂"


class FluxFP8Converter:
    """Utility helpers to convert FLUX bf16/fp16 checkpoints into scaled FP8 format."""

    def __init__(self, headless: bool, config: KohyaSSGUIConfig) -> None:
        self.headless = headless
        self.config = config or {}

    def _config_value(self, key: str, default):
        try:
            return self.config.get(key, default)
        except AttributeError:
            return default

    @staticmethod
    def _normalize_path(path: str) -> str:
        return os.path.abspath(os.path.expanduser(path.strip()))

    @staticmethod
    def _is_supported_file(path: str) -> bool:
        return path.lower().endswith(SUPPORTED_EXTENSIONS)

    def _infer_output_path(self, input_path: str, output_path: str = "") -> str:
        default_base, default_ext = os.path.splitext(input_path)
        default_name = f"{default_base}_FP8_scaled{default_ext}"

        if not output_path:
            return default_name

        normalized = self._normalize_path(output_path)
        if os.path.isdir(normalized):
            os.makedirs(normalized, exist_ok=True)
            return os.path.join(
                normalized, os.path.basename(default_name)
            )

        parent = os.path.dirname(normalized)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        return normalized

    def is_already_fp8_scaled(self, model_path: str) -> bool:
        try:
            with MemoryEfficientSafeOpen(model_path) as reader:
                metadata = reader.metadata() or {}
                if metadata.get("fp8_scaled", "").lower() == "true":
                    return True
                for key in reader.keys():
                    if key.endswith(".scale_weight"):
                        return True
        except Exception as err:
            log.warning(f"Could not inspect {model_path}: {err}")
        return False

    @staticmethod
    def _available_fp8_dtype():
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            raise RuntimeError(
                "torch.float8_e4m3fn is not available in this PyTorch build. "
                "Please upgrade to PyTorch 2.1+ with float8 support."
            )
        return fp8_dtype

    def _convert_model_to_fp8(
        self,
        input_path: str,
        output_path: str,
        quantization_mode: str,
        block_size: int,
        delete_original: bool,
    ) -> Tuple[bool, str]:
        if not input_path or not input_path.strip():
            return False, "Please provide an input model path."

        input_path = self._normalize_path(input_path)
        if not os.path.isfile(input_path):
            return False, f"Input file does not exist: {input_path}"

        if not self._is_supported_file(input_path):
            return False, "Unsupported file type. Use .safetensors, .pt, or .pth files."

        output_path = self._infer_output_path(input_path, output_path)
        if os.path.abspath(input_path) == os.path.abspath(output_path):
            return False, "Output path must be different from input path."

        if self.is_already_fp8_scaled(input_path):
            return False, "Model already appears to be FP8 scaled (contains scale_weight tensors)."

        try:
            fp8_dtype = self._available_fp8_dtype()
        except RuntimeError as err:
            return False, str(err)

        metadata = {}
        try:
            with MemoryEfficientSafeOpen(input_path) as reader:
                metadata = reader.metadata() or {}
        except Exception as err:
            log.warning(f"Could not read metadata from {input_path}: {err}")

        quant_mode = (quantization_mode or "tensor").strip().lower()
        if quant_mode not in {"tensor", "channel", "block"}:
            quant_mode = "tensor"

        if quant_mode == "block" and block_size < 16:
            block_size = 64

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(
            f"Converting '{input_path}' -> '{output_path}' "
            f"(quantization={quant_mode}, block_size={block_size}, device={device})"
        )

        try:
            fp8_state_dict = load_safetensors_with_fp8_optimization(
                model_files=[input_path],
                calc_device=device,
                target_layer_keys=FLUX_FP8_TARGET_KEYS,
                exclude_layer_keys=FLUX_FP8_EXCLUDE_KEYS,
                exp_bits=4,
                mantissa_bits=3,
                move_to_device=False,
                quantization_mode=quant_mode,
                block_size=block_size,
            )
        except Exception as err:
            log.error(f"FP8 optimization failed: {err}")
            return False, f"FP8 optimization failed: {err}"

        fp8_state_dict["scaled_fp8"] = torch.zeros(2, dtype=fp8_dtype)

        updated_metadata = metadata.copy()
        updated_metadata.update(
            {
                "format": metadata.get("format", "pt"),
                "fp8_scaled": "true",
                "quantization_mode": quant_mode,
            }
        )
        if quant_mode == "block":
            updated_metadata["block_size"] = str(block_size)

        try:
            mem_eff_save_file(fp8_state_dict, output_path, metadata=updated_metadata)
            log.info(f"Saved FP8 scaled model to {output_path}")
        except Exception as err:
            log.error(f"Failed to save FP8 model: {err}")
            return False, f"Failed to save FP8 model: {err}"
        finally:
            del fp8_state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        if delete_original:
            try:
                os.remove(input_path)
                log.info(f"Deleted original file: {input_path}")
            except Exception as err:
                log.warning(f"Failed to delete {input_path}: {err}")

        return True, f"Successfully converted model. Saved to:\n{output_path}"

    def convert_single_file(
        self,
        input_path: str,
        output_path: str,
        quantization_mode: str,
        block_size: float,
        delete_original: bool,
    ) -> str:
        success, message = self._convert_model_to_fp8(
            input_path,
            output_path,
            quantization_mode,
            int(block_size or 64),
            delete_original,
        )
        prefix = "[OK]" if success else "[ERR]"
        return f"{prefix} {message}"

    def _gather_model_files(self, folder: str) -> List[str]:
        files: List[str] = []
        for ext in SUPPORTED_EXTENSIONS:
            pattern = os.path.join(folder, "**", f"*{ext}")
            files.extend(glob.glob(pattern, recursive=True))
        seen = set()
        ordered_files = []
        for file in files:
            if file not in seen:
                ordered_files.append(file)
                seen.add(file)
        return ordered_files

    def batch_convert_models(
        self,
        input_folder: str,
        output_folder: str,
        quantization_mode: str,
        block_size: float,
        delete_original: bool,
    ) -> str:
        if not input_folder or not input_folder.strip():
            return "[ERR] Please provide an input folder."

        input_folder = self._normalize_path(input_folder)
        if not os.path.isdir(input_folder):
            return f"[ERR] Input folder does not exist: {input_folder}"

        destination_root = (
            self._normalize_path(output_folder) if output_folder and output_folder.strip() else input_folder
        )
        os.makedirs(destination_root, exist_ok=True)

        model_files = self._gather_model_files(input_folder)
        if not model_files:
            return f"[ERR] No model files found in {input_folder}"

        quant_mode = (quantization_mode or "tensor").strip().lower()
        block_size = int(block_size or 64)

        summary_lines: List[str] = []
        success_count = 0
        failed_count = 0
        skipped_count = 0

        for model_file in model_files:
            filename = os.path.basename(model_file)
            if "_FP8_scaled" in filename or self.is_already_fp8_scaled(model_file):
                skipped_count += 1
                summary_lines.append(f"[SKIP] {filename} (already FP8 scaled, skipped)")
                continue

            rel_path = os.path.relpath(model_file, input_folder)
            candidate_output = os.path.join(destination_root, rel_path)
            output_path = self._infer_output_path(candidate_output, "")

            success, message = self._convert_model_to_fp8(
                model_file,
                output_path,
                quant_mode,
                block_size,
                delete_original,
            )

            if success:
                success_count += 1
                summary_lines.append(f"[OK] {filename} -> {os.path.basename(output_path)}")
            else:
                failed_count += 1
                summary_lines.append(f"[ERR] {filename}: {message}")

        summary = (
            "Flux FP8 batch conversion finished.\n"
            f"Success: {success_count}\n"
            f"Failed: {failed_count}\n"
            f"Skipped: {skipped_count}\n\n"
            + "\n".join(summary_lines)
        )
        status_icon = "[OK]" if failed_count == 0 else "[WARN]"
        return f"{status_icon} {summary}"


def _toggle_block_size(mode: str):
    return gr.update(visible=(mode == "block"))


def flux_fp8_converter_tab(headless: bool, config: KohyaSSGUIConfig) -> None:
    converter = FluxFP8Converter(headless=headless, config=config)

    default_models_dir = os.path.join(scriptdir, "models")

    with gr.Tab("Flux FP8 Converter"):
        gr.Markdown("### Flux FP8 Scaled Converter")
        gr.Markdown(
            "Convert bf16/fp16 FLUX checkpoints into scaled FP8 weights compatible with ComfyUI and Musubi FP8 loaders. "
            "The converter adds block-wise scaling tensors (`.scale_weight`) plus a `scaled_fp8` flag."
        )

        single_default_mode = config.get("flux_fp8_converter.quant_mode", "tensor")
        batch_default_mode = config.get("flux_fp8_converter.batch_quant_mode", "tensor")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Single Model Conversion")
                single_input = gr.Textbox(
                    label="Input FLUX model",
                    placeholder="Path to bf16/fp16 FLUX DiT checkpoint (.safetensors)",
                    value=config.get("flux_fp8_converter.single_input", ""),
                )
                single_input_button = gr.Button(
                    FOLDER_BUTTON, elem_id="open_folder_small", visible=(not headless)
                )
                single_input_button.click(
                    get_any_file_path,
                    inputs=[single_input],
                    outputs=[single_input],
                    show_progress=False,
                )

                single_output = gr.Textbox(
                    label="Output file or folder (optional)",
                    info="Leave empty to save next to the input model with _FP8_scaled suffix. "
                    "If a folder is provided, the converter recreates the relative path with the suffix.",
                    value=config.get("flux_fp8_converter.single_output", ""),
                )
                single_output_button = gr.Button(
                    FOLDER_BUTTON, elem_id="open_folder_small", visible=(not headless)
                )
                single_output_button.click(
                    get_any_file_path,
                    inputs=[single_output],
                    outputs=[single_output],
                    show_progress=False,
                )

                single_quant_mode = gr.Radio(
                    label="Quantization mode",
                    choices=["tensor", "channel", "block"],
                    value=single_default_mode,
                    info="Tensor mode is the only layout ComfyUI/SwarmUI load today. Channel and Block provide better fidelity but will NOT load until their FP8 patches land.",
                )
                single_block_size = gr.Slider(
                    label="Block size",
                    minimum=16,
                    maximum=256,
                    step=16,
                    value=config.get("flux_fp8_converter.block_size", 64),
                    info="Higher block sizes (e.g., 128/256) improve quality but use more VRAM/time; 64 is the balanced default.",
                    visible=single_default_mode == "block",
                )
                single_quant_mode.change(
                    _toggle_block_size,
                    inputs=[single_quant_mode],
                    outputs=[single_block_size],
                    show_progress=False,
                )

                single_delete_original = gr.Checkbox(
                    label="Delete original after successful conversion",
                    value=config.get("flux_fp8_converter.delete_original", False),
                )

                single_convert_button = gr.Button("Convert Model", variant="primary")
                single_status = gr.Textbox(
                    label="Conversion status",
                    lines=6,
                    max_lines=12,
                    interactive=False,
                )

                single_convert_button.click(
                    converter.convert_single_file,
                    inputs=[
                        single_input,
                        single_output,
                        single_quant_mode,
                        single_block_size,
                        single_delete_original,
                    ],
                    outputs=[single_status],
                    show_progress=True,
                )

            with gr.Column():
                gr.Markdown("#### Batch Folder Conversion")
                batch_input = gr.Textbox(
                    label="Input folder",
                    placeholder="Folder containing bf16/fp16 FLUX checkpoints",
                    value=config.get("flux_fp8_converter.batch_input", default_models_dir),
                )
                batch_input_button = gr.Button(
                    FOLDER_BUTTON, elem_id="open_folder_small", visible=(not headless)
                )
                batch_input_button.click(
                    get_folder_path,
                    outputs=[batch_input],
                    show_progress=False,
                )

                batch_output = gr.Textbox(
                    label="Output folder (optional)",
                    info="Defaults to the input folder. Folder structure is preserved and _FP8_scaled is appended to filenames.",
                    value=config.get("flux_fp8_converter.batch_output", ""),
                )
                batch_output_button = gr.Button(
                    FOLDER_BUTTON, elem_id="open_folder_small", visible=(not headless)
                )
                batch_output_button.click(
                    get_folder_path,
                    outputs=[batch_output],
                    show_progress=False,
                )

                batch_quant_mode = gr.Radio(
                    label="Quantization mode",
                    choices=["tensor", "channel", "block"],
                    value=batch_default_mode,
                    info="Tensor = current Comfy/Swarm compatibility. Channel/Block are experimental (fail to load until loaders support per-channel/block scales).",
                )
                batch_block_size = gr.Slider(
                    label="Block size",
                    minimum=16,
                    maximum=256,
                    step=16,
                    value=config.get("flux_fp8_converter.batch_block_size", 64),
                    info="Higher block sizes mean better FP8 fidelity but more VRAM/time during conversion; 64 is a safe default.",
                    visible=batch_default_mode == "block",
                )
                batch_quant_mode.change(
                    _toggle_block_size,
                    inputs=[batch_quant_mode],
                    outputs=[batch_block_size],
                    show_progress=False,
                )

                batch_delete_original = gr.Checkbox(
                    label="Delete originals after each successful conversion",
                    value=config.get("flux_fp8_converter.batch_delete_original", False),
                )

                batch_convert_button = gr.Button("Start Batch Conversion", variant="primary")
                batch_status = gr.Textbox(
                    label="Batch conversion log",
                    lines=18,
                    max_lines=30,
                    interactive=False,
                )

                batch_convert_button.click(
                    converter.batch_convert_models,
                    inputs=[
                        batch_input,
                        batch_output,
                        batch_quant_mode,
                        batch_block_size,
                        batch_delete_original,
                    ],
                    outputs=[batch_status],
                    show_progress=True,
                )

        gr.Markdown(
            "**Compatibility warning:** ComfyUI/SwarmUI currently only load FP8 checkpoints whose "
            "`scale_weight` tensors are scalars (tensor mode). Channel/block quantization produces "
            "per-channel or per-block scales like the Musubi pipelines, but those models will fail to load "
            "until their loaders add support."
        )
        gr.Markdown(
            "Tip: Always feed standard bf16/fp16 FLUX base checkpoints. Pre-quantized FP8 models are skipped automatically. "
            "Tensor mode matches the reference `flux_dev_fp8_scaled_diffusion_model.safetensors`. Only switch to channel/block once your target runtime advertises scaled FP8 support."
        )

