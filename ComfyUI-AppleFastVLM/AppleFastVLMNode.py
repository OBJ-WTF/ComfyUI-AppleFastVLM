import torch
import os
import numpy as np
from PIL import Image
import folder_paths
import gc

class AppleFastVLMNode:
    """
    Node ComfyUI pour Apple FastVLM
    Permet d'effectuer de l'inférence avec les modèles FastVLM d'Apple
    """
    
    def __init__(self):
        self.model = None
        self.current_model_path = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail."
                }),
                "model_path": ("STRING", {
                    "default": "checkpoints/llava-fastvithd_0.5b_stage3",
                    "multiline": False
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "max_tokens": ("INT", {
                    "default": 256,
                    "min": 1,
                    "max": 2048,
                    "step": 1
                }),
            },
            "optional": {
                "load_in_8bit": ("BOOLEAN", {"default": False}),
                "load_in_4bit": ("BOOLEAN", {"default": False}),
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate_text"
    CATEGORY = "AppleVLM/FastVLM"
    
    def unload_model(self):
        """Décharge le modèle de la mémoire"""
        if self.model is not None:
            try:
                if 'model' in self.model and hasattr(self.model['model'], 'cpu'):
                    self.model['model'].cpu()
                del self.model
                self.model = None
                self.current_model_path = None
                
                # Nettoyage agressif de la mémoire
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                
                print("Modèle déchargé de la mémoire")
            except Exception as e:
                print(f"Erreur lors du déchargement: {e}")
    
    def load_model(self, model_path, load_in_8bit=False, load_in_4bit=False, force_reload=False):
        """Charge le modèle FastVLM"""
        try:
            # Si force_reload est activé, décharger d'abord
            if force_reload and self.model is not None:
                self.unload_model()
            
            # Si le modèle est déjà chargé avec le même chemin, on ne recharge pas
            if self.model is not None and self.current_model_path == model_path:
                # Vérifier que le modèle est toujours valide
                try:
                    if hasattr(self.model['model'], 'device'):
                        # Test rapide que le modèle est accessible
                        _ = self.model['model'].device
                        print("Modèle déjà chargé et valide, réutilisation")
                        return self.model
                except:
                    print("Modèle en cache invalide, rechargement...")
                    self.unload_model()
            
            print(f"Chargement du modèle FastVLM depuis: {model_path}")
            
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            
            model_name = get_model_name_from_path(model_path)
            
            # Charger le modèle avec les options de quantization
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                load_8bit=load_in_8bit,
                load_4bit=load_in_4bit,
                device_map="auto"
            )
            
            self.model = {
                'model': model,
                'tokenizer': tokenizer,
                'image_processor': image_processor,
                'context_len': context_len
            }
            self.current_model_path = model_path
            
            print("Modèle FastVLM chargé avec succès!")
            return self.model
            
        except Exception as e:
            self.unload_model()
            raise RuntimeError(f"Erreur lors du chargement du modèle: {str(e)}")
    
    def preprocess_image(self, image):
        """
        Convertit l'image du format ComfyUI (tensor) au format PIL
        """
        # ComfyUI utilise des tensors de forme [B, H, W, C] avec des valeurs entre 0 et 1
        # On prend la première image du batch
        image = image[0]
        
        # Convertir en numpy array et de [0,1] à [0,255]
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        
        # Créer une image PIL
        pil_image = Image.fromarray(image_np)
        
        return pil_image
    
    def generate_text(self, image, prompt, model_path, temperature=0.7, 
                     max_tokens=256, load_in_8bit=False, load_in_4bit=False,
                     force_reload=False):
        """
        Génère du texte à partir d'une image et d'un prompt
        """
        try:
            # Charger le modèle si nécessaire
            model_dict = self.load_model(model_path, load_in_8bit, load_in_4bit, force_reload)
            
            # Prétraiter l'image
            pil_image = self.preprocess_image(image)
            
            # Importer les utilitaires nécessaires
            from llava.mm_utils import process_images, tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            
            # Construire un prompt simple et direct (sans template de conversation)
            if DEFAULT_IMAGE_TOKEN not in prompt:
                simple_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
            else:
                simple_prompt = prompt
                
            print(f"Prompt envoyé: {simple_prompt[:100]}...")
            
            # Préparer l'image pour le modèle
            image_tensor = process_images(
                [pil_image],
                model_dict['image_processor'],
                model_dict['model'].config
            )
            
            if type(image_tensor) is list:
                image_tensor = [img.to(model_dict['model'].device, dtype=torch.float16) 
                              for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model_dict['model'].device, dtype=torch.float16)
            
            # Créer un prompt simple et direct pour FastVLM
            # FastVLM préfère un format simple sans template complexe
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            
            # Construire un prompt direct
            if DEFAULT_IMAGE_TOKEN not in prompt:
                simple_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
            else:
                simple_prompt = prompt
                
            print(f"Prompt envoyé au modèle: {simple_prompt[:100]}...")
            
            # Tokenizer le prompt directement
            input_ids = tokenizer_image_token(
                simple_prompt,
                model_dict['tokenizer'],
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).to(model_dict['model'].device)
            
            # Générer la réponse avec de meilleurs paramètres
            with torch.inference_mode():
                output_ids = model_dict['model'].generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=temperature > 0.01,
                    temperature=max(temperature, 0.01) if temperature > 0 else None,
                    max_new_tokens=max_tokens,
                    min_new_tokens=1,
                    use_cache=True,
                    pad_token_id=model_dict['tokenizer'].pad_token_id,
                    eos_token_id=model_dict['tokenizer'].eos_token_id,
                    repetition_penalty=1.1,  # Évite les répétitions
                    top_p=0.9,  # Nucleus sampling
                    num_beams=1,  # Pas de beam search pour la vitesse
                )
            
            # Décoder la sortie
            full_output = model_dict['tokenizer'].decode(
                output_ids[0],
                skip_special_tokens=True
            ).strip()
            
            # Nettoyer la sortie des artefacts de conversation
            generated_text = full_output
            
            # Enlever les préfixes de conversation si présents
            prefixes_to_remove = [
                "USER:", "ASSISTANT:", "Assistant:", "Human:", 
                "Question:", "Answer:", "Response:",
                "USER :", "ASSISTANT :", "Assistant :"
            ]
            
            for prefix in prefixes_to_remove:
                if generated_text.startswith(prefix):
                    generated_text = generated_text[len(prefix):].strip()
            
            # Si la réponse commence par notre prompt, l'enlever
            prompt_clean = prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip()
            if generated_text.lower().startswith(prompt_clean.lower()):
                generated_text = generated_text[len(prompt_clean):].strip()
            
            # Enlever les patterns de dialogue fictif au début
            lines = generated_text.split('\n')
            clean_lines = []
            skip_dialogue = False
            
            for line in lines:
                # Détecter les lignes de dialogue fictif
                if any(line.strip().startswith(p) for p in ["USER:", "ASSISTANT:", "Question:", "Answer:"]):
                    skip_dialogue = True
                    continue
                if skip_dialogue and line.strip() and not any(line.strip().startswith(p) for p in ["USER:", "ASSISTANT:", "Question:", "Answer:"]):
                    skip_dialogue = False
                if not skip_dialogue:
                    clean_lines.append(line)
            
            if clean_lines:
                generated_text = '\n'.join(clean_lines).strip()
            
            # Dernier nettoyage : si ça commence encore par un greeting générique, le couper
            greetings = ["Hello! How can I help", "Hi! How can I assist", "Hello there"]
            for greeting in greetings:
                if generated_text.startswith(greeting):
                    # Prendre tout après la première phrase complète
                    sentences = generated_text.split('. ')
                    if len(sentences) > 1:
                        generated_text = '. '.join(sentences[1:]).strip()
                    break
            
            print(f"Texte généré: {generated_text}")
            
            # Nettoyage agressif après génération
            del image_tensor, input_ids, output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return (generated_text,)
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            # En cas d'erreur, essayer de décharger le modèle pour le prochain essai
            self.unload_model()
            
            return (error_msg,)


# Mappings nécessaires pour ComfyUI
NODE_CLASS_MAPPINGS = {
    "AppleFastVLM": AppleFastVLMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AppleFastVLM": "Apple FastVLM Vision Language Model"
}