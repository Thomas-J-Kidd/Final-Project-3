# AI Model Comparison Tools

This collection of scripts helps you compare different AI models and generate presentation-ready visuals. These tools allow you to:

1. Compare multiple AI models side-by-side
2. Zoom in on specific regions to highlight differences
3. Generate reports and presentation-quality images

## Available Tools

### 1. `compare_models.py`

This script automatically compares different AI models on the same input images and generates side-by-side comparisons.

**Usage:**
```bash
python compare_models.py [--output_dir OUTPUT_DIR] [--num_regions NUM_REGIONS] [--num_images NUM_IMAGES]
```

- `--output_dir`: Directory to save comparison images (default: "model_comparisons")
- `--num_regions`: Number of regions to compare per image (default: 3)
- `--num_images`: Maximum number of images to compare (default: 10)

### 2. `compare_models_presentation.py`

This enhanced script allows for customized comparisons with presentation-quality output.

**Usage:**
```bash
python compare_models_presentation.py [OPTIONS]
```

**Options:**
- `--output_dir`: Directory to save comparison images (default: "presentation_comparisons")
- `--image`: Specific image to process (without _compare.png)
- `--roi_x`, `--roi_y`: Coordinates of the region of interest
- `--roi_size`: Size of ROI (width and height, default: 100)
- `--zoom`: Zoom factor for the ROI (default: 2)
- `--title`: Custom title for the comparison image
- `--output_filename`: Custom filename for the output image
- `--preset`: Use preset ROIs for specific examples (choices: 'deblurring', 'denoising', 'best')

**Example with specific image and region:**
```bash
python compare_models_presentation.py --image GOPR0396_11_00_000050 --roi_x 625 --roi_y 150 --roi_size 120 --title "Fine Detail Comparison"
```

**Example with preset:**
```bash
python compare_models_presentation.py --preset deblurring --zoom 3
```

### 3. `generate_presentation_images.py`

This script enhances existing model comparisons with better formatting and labels for presentations.

**Usage:**
```bash
python generate_presentation_images.py
```

The script automatically:
1. Finds existing model comparison images
2. Enhances them with better titles and formatting
3. Creates a summary image showing highlights from all comparisons
4. Saves everything to the "presentation_images" directory

## Workflow for Creating Presentation Materials

1. **Generate basic comparisons:**
   ```bash
   python compare_models.py
   ```

2. **Enhance for presentation:**
   ```bash
   python generate_presentation_images.py
   ```

3. **View the HTML report:**
   - Open `model_comparison_report.html` in a web browser

4. **Use in your presentation:**
   - The images in `presentation_images/` are ready to be directly imported into PowerPoint or other presentation software
   - Each image is formatted in 16:9 ratio with clear annotations

## Understanding the Output Images

Each comparison image shows:

1. **Top row**: Full images with highlighted regions of interest (red boxes)
2. **Bottom row**: Zoomed-in view of the regions of interest

The models are displayed side-by-side, allowing direct visual comparison of:
- Fine detail preservation
- Edge sharpness
- Texture quality
- Noise reduction
- Color accuracy

## Adding New Models

To add new models to the comparison:

1. Edit `compare_models.py` or `compare_models_presentation.py`
2. Modify the `model_dirs` and `model_names` lists to include your new model
3. Run the scripts as described above

## Tips for Effective Presentations

1. **Focus on regions with clear differences:**
   - Use the zoomed regions to highlight where models perform differently
   - Texture areas, edges, and fine details usually show the most differences

2. **Customize titles for your audience:**
   - Use `--title` to create descriptive titles for each comparison

3. **Use the HTML report as a reference:**
   - The report contains explanations that can help structure your presentation

4. **Supplement visuals with metrics:**
   - The HTML report includes performance metrics you can reference in your presentation
