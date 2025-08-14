import {
    initUI, initCanvas, displayLayers,
    getLoadedImageName, showLoading, hideLoading
} from './canvas_ui.js';


function init() {
    initUI();
    initCanvas();
}


document.getElementById('segmentButton').addEventListener('click', async function () {
    const name = getLoadedImageName();

    if (!name) {
        alert('Please load an image first!');
        return;
    }

    // Create an object with the image name
    const data = { imageName: name };
    showLoading();

    try {
        const response = await fetch("/segment-sketch", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        console.log('Segmentation successful:', result);
        const layers = result.layers; // Assuming layers is an array of URLs
        displayLayers(layers);

        // Show success message
        console.log('Segmentation completed successfully! Original drawing cleared. You can now draw on the segmented layers if desired.');

    } catch (error) {
        console.error('Error during segmentation:', error);
        alert(`Segmentation failed: ${error.message}`);
    } finally {
        hideLoading();
    }
});

document.addEventListener('DOMContentLoaded', function () {
    const segmentDrawingBtn = document.getElementById('segmentDrawingBtn');
    if (segmentDrawingBtn) {
        segmentDrawingBtn.addEventListener('click', async function () {
            const name = getLoadedImageName();

            if (!name || !name.startsWith('drawing_')) {
                alert('Please save a drawing first before segmenting!');
                return;
            }

            // Use the same segmentation logic
            document.getElementById('segmentButton').click();
        });
    }
});

init();