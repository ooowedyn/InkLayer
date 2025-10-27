import {
    initUI, initCanvas, displayLayers,
    getLoadedImageName, showLoading, hideLoading,
    getSelectedLayer, images, addImagesToDiv, redrawCanvas,
    getSelectedLayerUrl
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

// [Transform Layer] 버튼 클릭 이벤트
document.getElementById('inpaintGenerateBtn').addEventListener('click', async function () {
    const prompt = document.getElementById('inpaintPrompt').value.trim();
    const selectedLayer = getSelectedLayer();
    const imageName = getLoadedImageName();

    if (!selectedLayer) {
        alert("Please select a layer first!");
        return;
    }
    if (!imageName) {
        alert("Please load or segment an image first!");
        return;
    }
    if (!prompt) {
        alert("Please enter a description for transformation!");
        return;
    }

    // 선택된 레이어의 실제 URL 가져오기 (displayLayers 단계에서 저장된 정보)
    // 여기서는 image.src를 직접 참조
    const layerUrl = getSelectedLayerUrl();
    if (!layerUrl) {
        alert("Cannot find layer image path.");
        return;
    }

    showLoading();

    try {
        const response = await fetch("/inpaint", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image_name: imageName,
                layer_id: selectedLayer,
                layer_path: layerUrl,
                prompt: prompt
            }),
        });

        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Inpainting failed.");

        console.log("✅ Inpainting result:", result);

        // 응답받은 새 URL로 선택한 레이어 이미지 갱신
        if (images[selectedLayer]) {
            images[selectedLayer].src = result.layer_url;
            redrawCanvas(); // Canvas 갱신
        }

        // Layers UI 갱신
        const layersContainer = document.getElementById('strokeLayersDisplayContainer');
        layersContainer.innerHTML = '';
        addImagesToDiv(images, layersContainer);

        alert("Layer transformation completed successfully!");

    } catch (error) {
        console.error("❌ Error during inpainting:", error);
        alert(`Inpainting failed: ${error.message}`);
    } finally {
        hideLoading();
    }
});

init();