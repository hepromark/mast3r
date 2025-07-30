# MASt3R Parameters Documentation

This document explains all the parameters used in the MASt3R model and their default values from the Gradio interface.

## Parameter Categories

### 1. **Optimization Parameters**

These control the optimization process during 3D reconstruction:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lr1` | 0.07 | 0.01 - 0.2 | Coarse learning rate for initial alignment |
| `niter1` | 300 | 0 - 1000 | Number of iterations for coarse alignment |
| `lr2` | 0.01 | 0.005 - 0.05 | Fine learning rate for refinement |
| `niter2` | 300 | 0 - 1000 | Number of iterations for refinement |
| `optim_level` | 'refine+depth' | ['coarse', 'refine', 'refine+depth'] | Optimization level |

**Optimization Levels:**
- `coarse`: Only coarse alignment (sets niter2=0)
- `refine`: Coarse + fine alignment without depth optimization
- `refine+depth`: Full optimization including depth refinement

### 2. **Matching Parameters**

These control the feature matching process:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `matching_conf_thr` | 0.0 | 0.0 - 30.0 | Matching confidence threshold |
| `shared_intrinsics` | False | Boolean | Use shared intrinsics for all views |

### 3. **Scene Graph Parameters**

These control how image pairs are created:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `scenegraph_type` | 'complete' | ['complete', 'swin', 'logwin', 'oneref', 'retrieval'] | Scene graph type |
| `winsize` | 1 | 1 - N | Window size for sliding window |
| `win_cyclic` | False | Boolean | Use cyclic sequence |
| `refid` | 0 | 0 - N-1 | Reference image ID |

**Scene Graph Types:**
- `complete`: All possible image pairs
- `swin`: Sliding window approach
- `logwin`: Sliding window with long range connections
- `oneref`: Match one image with all others
- `retrieval`: Connect views based on similarity (requires retrieval model)

### 4. **Output Parameters**

These control the final 3D model generation:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `min_conf_thr` | 1.5 | 0.0 - 10.0 | Minimum confidence threshold for points |
| `as_pointcloud` | True | Boolean | Output as pointcloud instead of mesh |
| `mask_sky` | False | Boolean | Mask sky regions |
| `clean_depth` | True | Boolean | Clean up depthmaps |
| `transparent_cams` | False | Boolean | Make cameras transparent |
| `cam_size` | 0.2 | 0.001 - 1.0 | Camera size in visualization |
| `TSDF_thresh` | 0.0 | 0.0 - 1.0 | TSDF threshold for mesh generation |

## Usage Examples

### Basic Usage
```python
from mast3r_pipeline import MASt3RPipeline

pipeline = MASt3RPipeline()
pipeline.update_params(
    optim_level='refine+depth',
    min_conf_thr=2.0,
    as_pointcloud=True
)
```

### Advanced Usage with Custom Parameters
```python
pipeline = MASt3RPipeline()

# High-quality reconstruction
pipeline.update_params(
    lr1=0.05,              # Lower learning rate for stability
    niter1=500,            # More iterations
    lr2=0.005,             # Very fine refinement
    niter2=500,            # More refinement iterations
    optim_level='refine+depth',
    min_conf_thr=2.5,      # Higher confidence threshold
    as_pointcloud=False,    # Generate mesh instead
    clean_depth=True,
    cam_size=0.1           # Smaller cameras
)

# Fast reconstruction
pipeline.update_params(
    lr1=0.1,               # Higher learning rate
    niter1=100,            # Fewer iterations
    lr2=0.02,              # Coarser refinement
    niter2=100,            # Fewer refinement iterations
    optim_level='coarse',   # Only coarse alignment
    min_conf_thr=1.0,      # Lower confidence threshold
    as_pointcloud=True      # Pointcloud is faster
)
```

### Scene Graph Examples

**Complete Graph (default):**
```python
pipeline.update_params(scenegraph_type='complete')
```

**Sliding Window:**
```python
pipeline.update_params(
    scenegraph_type='swin',
    winsize=3,
    win_cyclic=False
)
```

**Retrieval-based (requires retrieval model):**
```python
pipeline.set_retrieval_model("path/to/retrieval/model")
pipeline.update_params(
    scenegraph_type='retrieval',
    winsize=20,    # Number of key images
    refid=10       # Number of neighbors
)
```

## Parameter Relationships

### Learning Rate Guidelines
- **lr1**: 0.05-0.1 for stable reconstruction, 0.1-0.2 for faster processing
- **lr2**: 0.005-0.02 for fine refinement, higher values may cause instability

### Iteration Guidelines
- **niter1**: 200-500 for good results, 100-200 for speed
- **niter2**: 200-500 for refinement, 0 for coarse-only

### Confidence Threshold Guidelines
- **min_conf_thr**: 1.0-2.0 for dense pointclouds, 2.0-3.0 for cleaner results
- Higher values = fewer but more reliable points
- Lower values = more points but potentially noisy

### Scene Graph Guidelines
- **complete**: Best for small datasets (<10 images)
- **swin/logwin**: Good for sequential images
- **retrieval**: Best for unordered image collections

## Troubleshooting

### Common Issues and Solutions

1. **Poor reconstruction quality:**
   - Increase `niter1` and `niter2`
   - Lower `lr1` and `lr2`
   - Increase `min_conf_thr`

2. **Slow processing:**
   - Use `optim_level='coarse'`
   - Reduce `niter1` and `niter2`
   - Use `as_pointcloud=True`

3. **Memory issues:**
   - Reduce `image_size` to 224
   - Use `optim_level='coarse'`
   - Process fewer images at once

4. **Noisy pointcloud:**
   - Increase `min_conf_thr`
   - Enable `clean_depth=True`
   - Use `mask_sky=True` if applicable

## Integration with Gradio

The parameters in this pipeline exactly match the Gradio interface defaults:

```python
# Gradio UI parameters (from mast3r/demo.py)
lr1 = gradio.Slider(label="Coarse LR", value=0.07, minimum=0.01, maximum=0.2, step=0.01)
niter1 = gradio.Slider(value=300, minimum=0, maximum=1000, step=1)
lr2 = gradio.Slider(label="Fine LR", value=0.01, minimum=0.005, maximum=0.05, step=0.001)
niter2 = gradio.Slider(value=300, minimum=0, maximum=1000, step=1)
optim_level = gradio.Dropdown(["coarse", "refine", "refine+depth"], value='refine+depth')
# ... etc
```

This ensures that your pipeline behaves identically to the Gradio interface when using the same parameters. 