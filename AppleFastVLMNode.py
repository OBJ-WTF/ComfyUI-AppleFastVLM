import os
import gc
import re
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from PIL import Image
import folder_paths


# === Helpers ================================================================

FASTVLM_SUBFOLDERS = [
    "models/FastVLM",
    "checkpoints",
]


def _list_fastvlm_checkpoints() -> List[str]:
    """Scan des dossiers habituels et retourne une liste de chemins de modèles.
    On liste les dossiers qui contiennent les fichiers attendus (p. ex. config.json, model.safetensors).
    """
    candidates: List[str] = []
    roots = [folder_paths.models_dir] + [
        os.path.join(folder_paths.models_dir, sf) for sf in FASTVLM_SUBFOLDERS
    ]

    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if not os.path.isdir(full):
                continue
            # Heuristique simple: présence de config.json OU *.safetensors
            has_cfg = os.path.isfile(os.path.join(full, "config.json"))
            has_st = any(f.endswith(".safetensors") for f in os.listdir(full))
            if has_cfg or has_st:
                if full not in seen:
                    candidates.append(full)
                    seen.add(full)

    # Fallback par défaut si rien trouvé
    if not candidates:
        candidates.append("checkpoints/llava-fastvithd_0.5b_stage3")
    return sorted(candidates)


def _can_bitsandbytes() -> bool:
    try:
        import bitsandbytes as bnb  # noqa: F401
        return True
    except Exception:
        return False


def _device_string() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_uint8_rgb(arr: np.ndarray) -> Image.Image:
    # Clamp, cast, drop alpha if present
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr, mode="RGB")


# === Node ==================================================================

class AppleFastVLMNode:
    """Node ComfyUI pour Apple FastVLM — version optimisée.

    Améliorations clés:
    - Découverte automatique des checkpoints (dropdown)
    - Détection bitsandbytes (fallback propre si non dispo)
    - Prétraitement image robuste (RGBA, clamp)
    - Gestion pad/eos id sûre, sampling plus stable
    - Option de post-traitement désactivable
    - Déchargement mémoire plus sûr
    """

    # Cache de modèle au niveau classe pour réutilisation inter-instances
    _cached: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        self.current_key: str | None = None

    @classmethod
    def INPUT_TYPES(cls):
        models = _list_fastvlm_checkpoints()
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "Describe this image in detail."},
                ),
                # Dropdown sur les modèles détectés, tout en acceptant une saisie libre
                "model_path": (
                    "STRING",
                    {
                        "default": models[0] if models else "checkpoints/llava-fastvithd_0.5b_stage3",
                        "choices": tuple(models),
                    },
                ),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 2048, "step": 1}),
            },
            "optional": {
                "load_in_8bit": ("BOOLEAN", {"default": False}),
                "load_in_4bit": ("BOOLEAN", {"default": False}),
                "force_reload": ("BOOLEAN", {"default": False}),
                "clean_output": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate_text"
    CATEGORY = "AppleVLM/FastVLM"

    # ------------------------------------------------------------------
    def _load_model(self, model_path: str, load_in_8bit: bool, load_in_4bit: bool, force_reload: bool) -> Dict[str, Any]:
        key = f"{model_path}|int8={load_in_8bit}|int4={load_in_4bit}"

        if force_reload and key in self._cached:
            self._unload_key(key)

        # Reuse cache
        if key in self._cached:
            try:
                _ = self._cached[key]["model"].device
                self.current_key = key
                print("[FastVLM] Reuse cached model:", key)
                return self._cached[key]
            except Exception:
                print("[FastVLM] Cached model invalid — reloading…")
                self._unload_key(key)

        print(f"[FastVLM] Loading: {model_path} (int8={load_in_8bit}, int4={load_in_4bit}) on {_device_string()}")

        # bitsandbytes check
        if (load_in_8bit or load_in_4bit) and not _can_bitsandbytes():
            print("[FastVLM] bitsandbytes not found. Falling back to full precision.")
            load_in_8bit = False
            load_in_4bit = False

        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            load_8bit=load_in_8bit,
            load_4bit=load_in_4bit,
            device_map="auto",
        )

        # ids spéciaux robustes
        pad_id = getattr(tokenizer, "pad_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0

        bundle = {
            "model": model,
            "tokenizer": tokenizer,
            "image_processor": image_processor,
            "context_len": context_len,
            "pad_id": int(pad_id),
            "eos_id": int(eos_id) if eos_id is not None else None,
        }
        self._cached[key] = bundle
        self.current_key = key
        print("[FastVLM] Loaded successfully.")
        return bundle

    # ------------------------------------------------------------------
    @classmethod
    def _unload_key(cls, key: str):
        try:
            pkt = cls._cached.pop(key, None)
            if pkt and "model" in pkt:
                try:
                    if hasattr(pkt["model"], "cpu"):
                        pkt["model"].cpu()
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            print(f"[FastVLM] Unloaded: {key}")
        except Exception as e:
            print(f"[FastVLM] Unload error for {key}: {e}")

    def unload_model(self):  # exposé pour l’utilisateur via le node, si besoin
        if self.current_key:
            self._unload_key(self.current_key)
            self.current_key = None

    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_image(image_tensor) -> Image.Image:
        # ComfyUI: [B, H, W, C] float32 in [0,1]
        img = image_tensor[0].detach().cpu().numpy()  # HWC
        return _to_uint8_rgb(img)

    # ------------------------------------------------------------------
    def generate_text(
        self,
        image,
        prompt: str,
        model_path: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        force_reload: bool = False,
        clean_output: bool = True,
    ) -> Tuple[str]:
        try:
            mdl = self._load_model(model_path, load_in_8bit, load_in_4bit, force_reload)

            pil_image = self._preprocess_image(image)

            from llava.mm_utils import process_images, tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

            # Assurer la présence du jeton image au début
            user_prompt = prompt if DEFAULT_IMAGE_TOKEN in prompt else f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"

            # Images -> tenseur
            img_t = process_images([pil_image], mdl["image_processor"], mdl["model"].config)
            if isinstance(img_t, list):
                img_t = [x.to(mdl["model"].device, dtype=torch.float16) for x in img_t]
            else:
                img_t = img_t.to(mdl["model"].device, dtype=torch.float16)

            # Tokenisation
            input_ids = (
                tokenizer_image_token(user_prompt, mdl["tokenizer"], IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .to(mdl["model"].device)
            )

            do_sample = temperature is not None and temperature > 0.01
            gen_kwargs = dict(
                images=img_t,
                do_sample=do_sample,
                temperature=max(float(temperature), 0.01) if do_sample else None,
                max_new_tokens=int(max_tokens),
                min_new_tokens=1,
                use_cache=True,
                pad_token_id=mdl["pad_id"],
                eos_token_id=mdl["eos_id"],
                repetition_penalty=1.1,
                top_p=0.9,
                num_beams=1,
            )

            with torch.inference_mode():
                output_ids = mdl["model"].generate(input_ids, **gen_kwargs)

            text = mdl["tokenizer"].decode(output_ids[0], skip_special_tokens=True).strip()

            if clean_output:
                text = self._post_clean(text, prompt)

            # GC soft
            del img_t, input_ids, output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print("[FastVLM] Generated text:", text[:120].replace("\n", " "))
            return (text,)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.unload_model()
            return (f"Erreur lors de la génération: {e}",)

    # ------------------------------------------------------------------
    @staticmethod
    def _post_clean(text: str, original_prompt: str) -> str:
        # 1) Retirer prompt recopié en tête (robuste, insensible à la casse)
        p = re.escape(original_prompt.strip())
        text = re.sub(rf"^\s*{p}\s*", "", text, flags=re.IGNORECASE)

        # 2) Retirer labels de dialogue courants en tête de ligne
        prefixes = [
            "USER:", "ASSISTANT:", "Assistant:", "Human:",
            "Question:", "Answer:", "Response:",
            "USER :", "ASSISTANT :", "Assistant :",
        ]
        # supprimer seulement au début
        for pr in prefixes:
            if text.startswith(pr):
                text = text[len(pr):].lstrip()

        # 3) Nettoyer un éventuel salut générique
        greetings = ["Hello!", "Hi!", "Hello there"]
        for g in greetings:
            if text.startswith(g):
                # couper la première phrase
                parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
                if len(parts) == 2:
                    text = parts[1].lstrip()
                break
        return text


# Mappings ComfyUI -----------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "AppleFastVLM": AppleFastVLMNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AppleFastVLM": "Apple FastVLM Vision Language Model",
}
