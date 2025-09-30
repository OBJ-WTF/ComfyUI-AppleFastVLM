# ComfyUI Apple FastVLM Node

Node personnalisé pour ComfyUI permettant d'utiliser les modèles Apple FastVLM pour la génération de texte à partir d'images.

## Description

FastVLM est un Vision Language Model (VLM) développé par Apple, optimisé pour une inférence rapide avec des images haute résolution. Ce node permet d'intégrer FastVLM directement dans vos workflows ComfyUI.

### Caractéristiques principales

- **Rapide**: 85x plus rapide que LLaVA-OneVision pour le Time-to-First-Token (TTFT)
- **Efficace**: Encoder de vision hybride produisant moins de tokens
- **Multiple variantes**: Support des modèles 0.5B, 1.5B et 7B
- **Quantization**: Support de la quantization 4-bit et 8-bit pour réduire l'utilisation mémoire

## Installation

### 1. Cloner le repository dans ComfyUI

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/votre-username/ComfyUI-AppleFastVLM.git
cd ComfyUI-AppleFastVLM
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Installer FastVLM

Cloner et installer le repository FastVLM d'Apple :

```bash
cd ..
git clone https://github.com/apple/ml-fastvlm.git
cd ml-fastvlm
pip install -e .
```

### 4. Télécharger les modèles

Les modèles pré-entraînés sont disponibles sur le site d'Apple. Vous pouvez les télécharger automatiquement :

```bash
bash get_models.sh
```

Ou télécharger manuellement un modèle spécifique :

**Modèles disponibles:**
- **FastVLM-0.5B** (léger, rapide): [Stage 3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip)
- **FastVLM-1.5B** (équilibré): [Stage 3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3.zip)
- **FastVLM-7B** (précis): [Stage 3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage3.zip)

Décompressez les modèles dans le dossier `checkpoints/`.

## Utilisation

### Dans ComfyUI

1. Lancez ComfyUI
2. Ouvrez votre navigateur sur l'interface ComfyUI
3. Faites un clic droit dans l'espace vide
4. Allez dans `Add Node` → `AppleVLM` → `FastVLM`
5. Ajoutez le node "Apple FastVLM Vision Language Model"

### Configuration du node

Le node accepte les paramètres suivants :

#### Entrées requises:

- **image**: Image à analyser (connexion depuis un node d'image ComfyUI)
- **prompt**: Texte décrivant ce que vous voulez que le modèle fasse
  - Exemple: "Describe this image in detail"
  - Exemple: "What objects can you see in this image?"
  - Exemple: "Is there a person in this image? If yes, what are they doing?"
  
- **model_path**: Chemin vers le modèle téléchargé
  - Par défaut: `checkpoints/llava-fastvithd_0.5b_stage3`
  
- **temperature**: Contrôle la créativité des réponses (0.0 à 2.0)
  - 0.0 = Déterministe, répétable
  - 0.2 = Peu de variabilité (recommandé pour la description)
  - 1.0 = Plus créatif et varié
  
- **max_tokens**: Longueur maximale de la réponse générée (1 à 2048)
  - Par défaut: 512

#### Entrées optionnelles:

- **load_in_8bit**: Active la quantization 8-bit (économise de la mémoire)
- **load_in_4bit**: Active la quantization 4-bit (économise encore plus de mémoire)

### Sorties:

- **generated_text**: Texte généré par le modèle

## Exemple de workflow

```
[Load Image] → [Apple FastVLM] → [Show Text]
                      ↑
                  [prompt: "Describe this image"]
```

## Configuration système recommandée

### Pour FastVLM-0.5B:
- GPU: 4GB+ VRAM
- RAM: 8GB+

### Pour FastVLM-1.5B:
- GPU: 8GB+ VRAM  
- RAM: 16GB+

### Pour FastVLM-7B:
- GPU: 16GB+ VRAM (ou 8GB avec quantization 4-bit)
- RAM: 24GB+

## Conseils d'utilisation

### Prompts efficaces

- **Description générale**: "Describe this image in detail"
- **Questions spécifiques**: "How many people are in this image?"
- **Identification d'objets**: "List all the objects you can see in this image"
- **Analyse de scène**: "What is happening in this image?"
- **OCR**: "What text can you see in this image?"

### Optimisation des performances

1. **Utiliser la quantization**: Activez `load_in_4bit` pour les gros modèles
2. **Choisir le bon modèle**: 
   - 0.5B pour la vitesse
   - 7B pour la précision
3. **Réduire max_tokens**: Si vous n'avez besoin que de réponses courtes
4. **Temperature basse**: Pour des résultats plus consistants

## Dépannage

### Erreur "Out of memory"
- Activez la quantization 4-bit ou 8-bit
- Utilisez un modèle plus petit (0.5B au lieu de 7B)
- Fermez les autres applications consommant de la VRAM

### Le modèle ne se charge pas
- Vérifiez que le chemin `model_path` est correct
- Assurez-vous que le modèle est bien décompressé
- Vérifiez que toutes les dépendances sont installées

### Réponses de mauvaise qualité
- Essayez un modèle plus grand (7B)
- Ajustez la temperature (essayez 0.7-1.0)
- Reformulez votre prompt pour être plus spécifique

## Structure du projet

```
ComfyUI-AppleFastVLM/
├── AppleFastVLMNode.py    # Code principal du node
├── __init__.py            # Point d'entrée pour ComfyUI
├── requirements.txt       # Dépendances Python
└── README.md             # Ce fichier
```

## Licence

Ce node utilise le code de FastVLM d'Apple. Consultez les licences suivantes:
- [LICENSE FastVLM](https://github.com/apple/ml-fastvlm/blob/main/LICENSE)
- [LICENSE_MODEL](https://github.com/apple/ml-fastvlm/blob/main/LICENSE_MODEL)

## Citation

Si vous utilisez FastVLM dans vos travaux, citez:

```bibtex
@InProceedings{fastvlm2025,
    author = {Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari},
    title = {FastVLM: Efficient Vision Encoding for Vision Language Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2025},
}
```

## Crédits

- **FastVLM**: Apple Inc.
- **LLaVA**: Haotian Liu et al.
- **ComfyUI**: comfyanonymous

## Support

Pour les problèmes liés au node ComfyUI, ouvrez une issue sur le repository.
Pour les problèmes liés à FastVLM, consultez le [repository officiel](https://github.com/apple/ml-fastvlm).