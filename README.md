#!/bin/bash
# Quick Start Script for Setting Up GitHub Repository
# Copy and paste these commands one by one

echo "=== ComfyUI-AppleFastVLM GitHub Setup ==="
echo ""

# Step 1: Make script executable
echo "Step 1: Setting file permissions..."
chmod +x get_models.sh
echo "✓ Done"
echo ""

# Step 2: Initialize Git
echo "Step 2: Initializing Git repository..."
git init
echo "✓ Git initialized"
echo ""

# Step 3: Add all files
echo "Step 3: Adding files to Git..."
git add .
echo "✓ Files added"
echo ""

# Step 4: First commit
echo "Step 4: Creating first commit..."
git commit -m "Initial commit: ComfyUI Apple FastVLM Node"
echo "✓ First commit created"
echo ""

# Step 5: Instructions for GitHub
echo "=== Next Steps (Manual) ==="
echo ""
echo "1. Go to: https://github.com"
echo "2. Click '+' → 'New repository'"
echo "3. Name: ComfyUI-AppleFastVLM"
echo "4. Public repository"
echo "5. DO NOT initialize with README"
echo "6. Click 'Create repository'"
echo ""
echo "Then run these commands (replace YOUR-USERNAME):"
echo ""
echo "git remote add origin https://github.com/YOUR-USERNAME/ComfyUI-AppleFastVLM.git"
echo "git branch -M main"
echo "git push -u origin main"
echo ""
echo "=== Optional: Create .github folder for issue templates ==="
echo "mkdir -p .github/ISSUE_TEMPLATE"
echo ""
echo "=== Optional: Create images folder ==="
echo "mkdir -p images"
echo ""
echo "=== After pushing, configure on GitHub: ==="
echo "1. Add topics: comfyui, vision-language-model, apple-fastvlm, etc."
echo "2. Create first release (v1.0.0)"
echo "3. Add screenshots to images/ folder"
echo ""
echo "🎉 Setup script complete! Follow the instructions above."
