"""
WSI Thumbnail Generation and Region Annotation Tool
Features:
1. Generate WSI thumbnails using OpenSlide
2. Select regions of interest by drawing polygons
3. Generate fixed-size grid coordinates within selected regions
4. Save grid coordinates to H5 file for further processing

Parameters:
- SCALE_FACTOR: Scaling factor controlling the ratio between thumbnail and original image
- rect_size: Grid size (pixels), automatically determined based on objective power
  - 1024 pixels for 40x magnification
  - 512 pixels for other magnifications
"""

import openslide
from PIL import Image
import cv2
import numpy as np
import h5py
import os

def process_wsi_slide():
    # Basic configuration
    SCALE_FACTOR = 100
    svs_file = 'TCGA-BR-6709-01Z-00-DX1.92df4063-8b47-4655-a010-edc385b35840.svs'
    output_image = 'TCGA_test_h5.png'

    # Step 1: Generate thumbnail
    print("Step 1: Generating WSI thumbnail...")
    slide = openslide.OpenSlide(svs_file)
    width, height = slide.dimensions
    
    # Get objective power
    try:
        objective_power = int(slide.properties.get('openslide.objective-power', 40))
    except (ValueError, TypeError):
        objective_power = 40  # Default value
        
    print(f"Objective power: {objective_power}x")
    
    # Determine grid size based on magnification
    rect_size = 1024 if objective_power == 40 else 512
    
    thumbnail_width = width // SCALE_FACTOR
    thumbnail_height = height // SCALE_FACTOR
    thumbnail = slide.get_thumbnail((thumbnail_width, thumbnail_height))
    thumbnail.save(output_image)
    
    print(f"Thumbnail generated: {output_image}")
    print(f"Grid size set to: {rect_size}px (based on {objective_power}x objective power)")
    
    # Step 2: Process coordinate annotation
    print("\nStep 2: Launching coordinate annotation system...")
    thumbnail_image = cv2.imread(output_image)
    if thumbnail_image is None:
        raise FileNotFoundError(f"Cannot find thumbnail file: {output_image}")
    
    clone = thumbnail_image.copy()
    points = []
    coords = []

    def draw_polygon(event, x, y, flags, param):
        nonlocal clone, points
        
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Added vertex #{len(points)}: ({x}, {y})")
            cv2.circle(clone, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow("Coordinate Drawing System", clone)

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(points) > 2:
                print(f"Processing polygon with {len(points)} vertices")
                
                polygon_original = [(x*SCALE_FACTOR, y*SCALE_FACTOR) for (x,y) in points]
                
                original_dims = (thumbnail_image.shape[0]*SCALE_FACTOR, 
                               thumbnail_image.shape[1]*SCALE_FACTOR)
                mask = np.zeros((int(original_dims[0]), int(original_dims[1])), dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(polygon_original, dtype=np.int32)], 255)

                for y_base in range(0, mask.shape[0] - rect_size, rect_size):
                    for x in range(0, mask.shape[1] - rect_size, rect_size):
                        center = (x + rect_size//2, y_base + rect_size//2)
                        if mask[center[1], center[0]] == 255:
                            y_end = y_base + rect_size
                            coords.append((x, y_end))
                            
                            scaled_rect = (
                                int(x/SCALE_FACTOR), 
                                int(y_base/SCALE_FACTOR),
                                int(rect_size/SCALE_FACTOR), 
                                int(rect_size/SCALE_FACTOR)
                            )
                            cv2.rectangle(clone, 
                                        (scaled_rect[0], scaled_rect[1]),
                                        (scaled_rect[0]+scaled_rect[2], 
                                         scaled_rect[1]+scaled_rect[3]),
                                        (0, 0, 255), 1)
                
                print(f"Region processing completed, added grids: {len(coords)}")
                cv2.imshow("Coordinate Drawing System", clone)
            else:
                print("At least 3 vertices are needed to define a polygon")
                
            points = []
            clone = thumbnail_image.copy()

    cv2.namedWindow("Coordinate Drawing System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Coordinate Drawing System", 1000, 800)
    cv2.setMouseCallback("Coordinate Drawing System", draw_polygon)
    cv2.imshow("Coordinate Drawing System", clone)

    print("="*60)
    print("Operation Guide:")
    print("1. Left-click to add polygon vertices")
    print("2. Right-click to complete current polygon and generate grids")
    print("3. Press any key to exit the system")
    print("="*60)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key != 255:
            break

    # Step 3: Save results
    print("\nStep 3: Saving processing results...")
    cv2.imwrite("annotated_thumbnail.png", clone)
    
    # Save coordinates and parameters
    with h5py.File("grid_coordinates.h5", "w") as h5file:
        # Save coordinates
        coordinates = np.array([(x, y) for (x, y) in coords], dtype=np.int32)
        h5file.create_dataset("coords", data=coordinates, compression="gzip")
        
        # Also save related parameters for future use
        h5file.create_dataset("scale_factor", data=SCALE_FACTOR)
        h5file.create_dataset("rect_size", data=rect_size)
        h5file.create_dataset("objective_power", data=objective_power)
        h5file.create_dataset("slide_dimensions", data=[width, height])
        
    print(f"Processing completed! Results saved:")
    print(f"- Annotated thumbnail: annotated_thumbnail.png")
    print(f"- Grid coordinates file: grid_coordinates.h5")
    print(f"- Total marked grids: {len(coords)}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_wsi_slide()
