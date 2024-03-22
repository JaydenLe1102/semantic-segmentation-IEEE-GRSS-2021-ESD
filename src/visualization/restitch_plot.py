import os
from pathlib import Path
from typing import List, Tuple
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.preprocessing.subtile_esd_hw02 import TileMetadata, Subtile, restitch



def get_subtile_from_datamodule(datamodule, tile_id):
    
    subTilesX = []
    subTilesY = []
    
    subTilesMetadata = []
    
    for x, y, m in datamodule.val_dataloader():
        for i in range(len(m)):
            tileMetadata = m[i]
            if tileMetadata.parent_tile_id == tile_id:
                subTilesX.append(x[i])
                subTilesY.append(y[i])
                subTilesMetadata.append(m[i])
                

    
    if len(subTilesX) == 0:
        for x, y, m in datamodule.train_dataloader():
            for i in range(len(m)):
                tileMetadata = m[i]
                if tileMetadata.parent_tile_id == tile_id:
                    subTilesX.append(x[i])
                    subTilesY.append(y[i])
                    subTilesMetadata.append(m[i])

    #sort the subTilesX and subTilesY by the "x_gt" and "y_gt" in subTilesMetadata
    
    # Combine the lists for sorting based on subTilesMetadata
    combined = zip(subTilesX, subTilesY, subTilesMetadata)

    # Sort based on 'x_gt' and 'y_gt' in subTilesMetadata
    sorted_combined = sorted(combined, key=lambda x: (x[2].x_gt, x[2].y_gt))

    # Unzip the sorted lists
    subTilesX, subTilesY, subTilesMetadata = zip(*sorted_combined)
    
    #return subTilesX, subTilesY, subTilesMetadata
    return torch.stack(subTilesX), torch.stack(subTilesY), subTilesMetadata      
    

def get_subtile_from_datamodule_testing(datamodule, tile_id):
    
    subTilesX = []
    subTilesY = []
    
    subTilesMetadata = []
    
    for x, y, m in datamodule.val_dataloader():
        for i in range(len(m)):
            tileMetadata = m[i]
            if tileMetadata.parent_tile_id == tile_id:
                subTilesX.append(x[i])
                subTilesY.append(y[i])
                subTilesMetadata.append(m[i])
                

    
    if len(subTilesX) == 0:
        for x, y, m in datamodule.train_dataloader():
            for i in range(len(m)):
                tileMetadata = m[i]
                if tileMetadata.parent_tile_id == tile_id:
                    subTilesX.append(x[i])
                    subTilesY.append(y[i])
                    subTilesMetadata.append(m[i])

    ##sort the subTilesX and subTilesY by the "x_gt" and "y_gt" in subTilesMetadata
    
    ## Combine the lists for sorting based on subTilesMetadata
    #combined = zip(subTilesX, subTilesY, subTilesMetadata)

    ## Sort based on 'x_gt' and 'y_gt' in subTilesMetadata
    #sorted_combined = sorted(combined, key=lambda x: (x[2].x_gt, x[2].y_gt))

    ## Unzip the sorted lists
    #subTilesX, subTilesY, subTilesMetadata = zip(*sorted_combined)
    
    ##return subTilesX, subTilesY, subTilesMetadata
    return torch.stack(subTilesX), torch.stack(subTilesY), subTilesMetadata 
    
    #return subTilesX, subTilesY, subTilesMetadata

def restitch_and_plot(options, datamodule, model, parent_tile_id, satellite_type="sentinel2", rgb_bands=[3,2,1], image_dir: None | str | os.PathLike = None):
    """
    Plots the 1) rgb satellite image 2) ground truth 3) model prediction in one row.

    Args:
        options: EvalConfig
        datamodule: ESDDataModule
        model: ESDSegmentation
        parent_tile_id: str
        satellite_type: str
        rgb_bands: List[int]
    """
    processed_dir = options.processed_dir / 'Val'
    
    stitched_image,stitched_ground_truth, y_pred = restitch_eval(processed_dir, satellite_type, parent_tile_id, (0, options.tile_size_gt), (0, options.tile_size_gt), datamodule, model)
    
    y_pred = np.argmax(y_pred, axis=0)
    stitched_ground_truth = stitched_ground_truth[0]
    rgb_img = np.dstack([stitched_image[0,i,:,:] for i in rgb_bands])

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

    # make sure to use cmap=cmap, vmin=-0.5 and vmax=3.5 when running
    # axs[i].imshow on the 1d images in order to have the correct 
    # colormap for the images.
    # On one of the 1d images' axs[i].imshow, make sure to save its output as 
    # `im`, i.e, im = axs[i].imshow
    
    # Plot RGB image
    axs[0].imshow(rgb_img)
    axs[0].set_title('RGB Image')

    # Plot ground truth with colormap
    im = axs[1].imshow(stitched_ground_truth, cmap=cmap, vmin=-0.5, vmax=3.5)  # Use correct colormap range
    axs[1].set_title('Ground Truth')

    # Plot prediction with colormap (assuming prediction_image holds class labels)
    axs[2].imshow(y_pred, cmap=cmap, vmin=-0.5, vmax=3.5)  # Use correct colormap range
    axs[2].set_title('Model Prediction')
    plt.tight_layout()
    
    # The following lines sets up the colorbar to the right of the images    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels(['Sttlmnts Wo Elec', 'No Sttlmnts Wo Elec', 'Sttlmnts W Elec', 'No Sttlmnts W Elec'])
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / f"restitched_visible_gt_predction_{parent_tile_id}.png")
        plt.close()

# helper function: this would iterate through either val_dataset or train_dataset to find the target tiles.
def find_tile(dataloader, id, i, j):
        for X, y, m in dataloader:
            for k in range(len(m)):
                tileMetadata = m[k]
                print(tileMetadata.parent_tile_id, id, tileMetadata.x_gt, i, tileMetadata.y_gt, j)
                if tileMetadata.parent_tile_id == id and tileMetadata.x_gt == i and tileMetadata.y_gt == j:
                    return X[i], y[i], m[k], True

        return None, None, None, False

def restitch_eval(dir: str | os.PathLike, satellite_type: str, tile_id: str, range_x: Tuple[int, int], range_y: Tuple[int, int], datamodule, model) -> np.ndarray:
    """
    Given a directory of processed subtiles, a tile_id and a satellite_type, 
    this function will retrieve the tiles between (range_x[0],range_y[0])
    and (range_x[1],range_y[1]) in order to stitch them together to their
    original image. It will also get the tiles from the datamodule, evaluate
    it with model, and stitch the ground truth and predictions together.

    Input:
        dir: str | os.PathLike
            Directory where the subtiles are saved
        satellite_type: str
            Satellite type that will be stitched
        tile_id: str
            Tile id that will be stitched
        range_x: Tuple[int, int]
            Range of tiles that will be stitched on width dimension [0,5)]
        range_y: Tuple[int, int]
            Range of tiles that will be stitched on height dimension
        datamodule: pytorch_lightning.LightningDataModule
            Datamodule with the dataset
        model: pytorch_lightning.LightningModule
            LightningModule that will be evaluated
    
    Output:
        stitched_image: np.ndarray
            Stitched image, of shape (time, bands, width, height)
        stitched_ground_truth: np.ndarray
            Stitched ground truth of shape (width, height)
        stitched_prediction_subtile: np.ndarray
            Stitched predictions of shape (width, height)
    """
    
    dir = Path(dir)
    satellite_subtile = []
    ground_truth_subtile = []
    predictions_subtile = []
    satellite_metadata_from_subtile = []
    
    subtilesX, subtilesY, subtilesMetaData = get_subtile_from_datamodule_testing(datamodule, tile_id)
    
    for i in range(*range_x):
        satellite_subtile_row = []
        ground_truth_subtile_row = []
        predictions_subtile_row = []
        satellite_metadata_from_subtile_row = []
        for j in range(*range_y):
            subtile = Subtile().load(dir / 'subtiles' / f"{tile_id}_{i}_{j}.npz")

            # find the tile in the datamodule
            #########################################################
            # below implementation is brute force and might not work at all
            # REMINDER: implement a function that would extract target tile from datamodule
            ##########################################################
            
            #X, y, m, found_val = find_tile(datamodule.val_dataloader(), tile_id, i, j)
            #if not found_val:
            #    #X, y, m, found_train = find_tile(datamodule.train_dataloader(), tile_id, i, j)
            #    #if not found_train:
            #    print(f"error: no matching tile found")
            
            #X = subtilesX[i]
            #y = subtilesY[i]
            #m = subtilesMetaData[i]
            
            for k in range(len(subtilesMetaData)):
                tileMetadata = subtilesMetaData[k]
                if tileMetadata.parent_tile_id == tile_id and tileMetadata.x_gt == i and tileMetadata.y_gt == j:
                    X = subtilesX[k]
                    y = subtilesY[k]
                    m = subtilesMetaData[k]
                    break
            

            ##########################################################
            # evaluate the tile with the model
            # You need to add a dimension of size 1 at dim 0 so that
            # some CNN layers work
            # i.e., (batch_size, channels, width, height) with batch_size = 1
            # make sure that the tile is in GPU memory, i.e., X = X.cuda()
            
            X = X.unsqueeze(0)
            #X = torch.tensor(X, device='mps:0')
            model.eval()
            #print("after squeeze")
            #print(X.shape)
            predictions = model(X)
            predictions = predictions.detach().cpu().numpy()
            predictions = predictions.squeeze(0)

            y = y.cpu().numpy()
            
            #print("shape of output")
            #print(predictions.shape)
            #print(y.shape)


            # convert y to numpy array
            # detach predictions from the gradient, move to cpu and convert to numpy

            ground_truth_subtile_row.append(y)
            predictions_subtile_row.append(predictions)
            satellite_subtile_row.append(subtile.satellite_stack[satellite_type])
            satellite_metadata_from_subtile_row.append(subtile.tile_metadata)
        ground_truth_subtile.append(np.concatenate(ground_truth_subtile_row, axis=-1))
        predictions_subtile.append(np.concatenate(predictions_subtile_row, axis=-1))
        satellite_subtile.append(np.concatenate(satellite_subtile_row, axis=-1))
        satellite_metadata_from_subtile.append(satellite_metadata_from_subtile_row)
    return np.concatenate(satellite_subtile, axis=-2), np.concatenate(ground_truth_subtile, axis=-2), np.concatenate(predictions_subtile, axis=-2)
