var canvas = document.getElementById('myCanvas');
var context = canvas.getContext('2d');

var images = {};
var imagePositions = {}
var selectedImage = null;
var selectedLayer = null; // for copy and delete
var dragOffsetX, dragOffsetY;
var loadedImageName = ""// if we upload sketch from UI, we will store the name here
var CANVAS_WIDTH = 570;
var CANVAS_HEIGHT = 570;
var snapshots = []

// Drawing variables
var isDrawing = false;
var drawingMode = false;
var brushSize = 5;
var brushColor = '#000000';
var drawingData = []; // Store drawing paths for undo functionality

/** DRAWING FUNCTIONS */
function toggleDrawingMode() {
    drawingMode = !drawingMode;
    const btn = document.getElementById('drawModeBtn');
    if (drawingMode) {
        btn.textContent = 'Exit Drawing Mode';
        btn.className = 'px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600';
        canvas.style.cursor = 'crosshair';
    } else {
        btn.textContent = 'Drawing Mode';
        btn.className = 'px-4 py-2 bg-blue-200 text-black rounded hover:bg-blue-600';
        canvas.style.cursor = 'default';
    }
}

function clearDrawing() {
    drawingData = [];
    redrawCanvas();
}

export function clearDrawingOnly() {
    // Clear only the drawing data, keep the images
    drawingData = [];
    redrawCanvas();
}

function updateBrushSize(event) {
    brushSize = event.target.value;
    document.getElementById('brushSizeDisplay').textContent = brushSize;
}

function updateBrushColor(event) {
    brushColor = event.target.value;
}

async function saveDrawing() {
    try {
        showLoading();

        // Convert canvas to blob
        const canvas = document.getElementById('myCanvas');
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));

        // Create FormData and append the blob
        const formData = new FormData();
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `drawing_${timestamp}.png`;
        formData.append('image', blob, filename);

        // Send to backend
        const response = await fetch('/upload-image', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (response.ok) {
            console.log('Drawing saved successfully:', result);
            loadedImageName = filename.split('.')[0];
            console.log('Drawing saved successfully! You can now segment it.');
        } else {
            throw new Error(result.error || 'Failed to save drawing');
        }

    } catch (error) {
        console.error('Error saving drawing:', error);
        alert('Failed to save drawing: ' + error.message);
    } finally {
        hideLoading();
    }
}


/** UI FUNCTIONS THAT INTERACTS WITH API */
export function initUI() {
    document.addEventListener('DOMContentLoaded', fetchDemosCallback);
    document.getElementById('imageLoader').onchange = imageLoaderCallback
    document.getElementById('deleteLayerButton').addEventListener('click', deleteBtnCallback)
    document.getElementById('copyLayerButton').addEventListener('click', copyLayerBtnCallback)
    // Drawing mode controls
    document.getElementById('drawModeBtn').addEventListener('click', toggleDrawingMode);
    document.getElementById('saveDrawingBtn').addEventListener('click', saveDrawing);
    document.getElementById('brushSize').addEventListener('input', updateBrushSize);
    document.getElementById('brushColor').addEventListener('change', updateBrushColor);
}

function resizeCanvas() {
    canvas.width = CANVAS_WIDTH
    canvas.height = CANVAS_HEIGHT
    drawImages();
}

export function initCanvas() {
    // Initial sizing
    resizeCanvas();

    // Add resize listener
    window.addEventListener('resize', resizeCanvas);

    // Modified mouse handlers
    canvas.onmousedown = canvasOnMouseDown;
    canvas.onmousemove = canvasOnMouseMove;
    canvas.onmouseup = canvasOnMouseUp;
    canvas.onmouseleave = canvasOnMouseUp; // Stop drawing when mouse leaves canvas
}


function startDrawing(x, y) {
    isDrawing = true;
    const currentPath = {
        points: [{ x, y }],
        color: brushColor,
        size: brushSize
    };
    drawingData.push(currentPath);

    context.beginPath();
    context.moveTo(x, y);
    context.strokeStyle = brushColor;
    context.lineWidth = brushSize;
    context.lineCap = 'round';
    context.lineJoin = 'round';
}

function continueDrawing(x, y) {
    if (!isDrawing) return;

    const currentPath = drawingData[drawingData.length - 1];
    currentPath.points.push({ x, y });

    context.lineTo(x, y);
    context.stroke();
}

function stopDrawing() {
    isDrawing = false;
    context.beginPath();
}

function redrawCanvas() {
    context.clearRect(0, 0, canvas.width, canvas.height);

    // Draw background images first
    drawImages();

    // Then draw all the drawing paths
    drawingData.forEach(path => {
        if (path.points.length > 0) {
            context.beginPath();
            context.strokeStyle = path.color;
            context.lineWidth = path.size;
            context.lineCap = 'round';
            context.lineJoin = 'round';

            context.moveTo(path.points[0].x, path.points[0].y);
            for (let i = 1; i < path.points.length; i++) {
                context.lineTo(path.points[i].x, path.points[i].y);
            }
            context.stroke();
        }
    });
}


/** UTILS FOR API CALLS */
export function getSnapshots() {
    return snapshots;
}

export function getLoadedImageName() {
    return loadedImageName;
}

export function showLoading() {
    document.getElementById('loading-spinner').style.display = 'block';
}

export function hideLoading() {
    document.getElementById('loading-spinner').style.display = 'none';
}

export function clearCanvasControls() {
    selectedImage = null;
    selectedLayer = null;
}

/** UI FUNCTIONS **/
function saveSnapshotCallback() {
    const snapshot = convertCanvasToBWImageUrl();
    snapshots.push(snapshot);

    // Create an image element to display the snapshot
    const img = document.createElement('img');
    img.src = snapshot;
    img.className = 'snapshot';
    img.width = 150;
    img.height = 150;
    img.className = "border border-2"
    const snapshotsDiv = document.getElementById('snapshots');
    snapshotsDiv.appendChild(img);
}

function drawImages() {
    // Only clear and redraw background images, not the drawing
    const tempCanvas = document.createElement('canvas');
    const tempContext = tempCanvas.getContext('2d');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;

    // Save current drawing
    tempContext.drawImage(canvas, 0, 0);

    // Clear main canvas
    context.clearRect(0, 0, canvas.width, canvas.height);

    // Use canvas size as the reference
    const scaleFactor = Math.min(canvas.width, canvas.height);

    for (var key in images) {
        if (images.hasOwnProperty(key)) {
            var pos = imagePositions[key];
            if (pos) {
                // Scale relative to canvas size
                const scaledPos = {
                    x: (pos.x / CANVAS_WIDTH) * scaleFactor,
                    y: (pos.y / CANVAS_HEIGHT) * scaleFactor,
                    width: (pos.width / CANVAS_WIDTH) * scaleFactor,
                    height: (pos.height / CANVAS_HEIGHT) * scaleFactor
                };

                context.drawImage(
                    images[key],
                    scaledPos.x,
                    scaledPos.y,
                    scaledPos.width,
                    scaledPos.height
                );
            }
        }
    }

    // Restore drawing on top
    context.drawImage(tempCanvas, 0, 0);
}

export function addNewLayer(imageUrl) {
    var newKey = Object.keys(images).length.toString();
    images[newKey] = new Image();
    images[newKey].onload = function () {
        imagePositions[newKey] = { x: 0, y: 0, width: CANVAS_WIDTH, height: CANVAS_HEIGHT };
        redrawCanvas();
        const layersContainer = document.getElementById('strokeLayersDisplayContainer');
        layersContainer.innerHTML = '';
        addImagesToDiv(images, layersContainer);
    };
    images[newKey].src = imageUrl;

    console.log('Added new layer with image src:', images[newKey].src)
}

export function displayLayers(imageUrls) {
    loadImages(imageUrls, function (loadedImages) {
        for (var key in loadedImages) {
            images[key] = loadedImages[key];
            imagePositions[key] = {
                x: 0,
                y: 0,
                width: CANVAS_WIDTH,  // Using consistent reference size
                height: CANVAS_HEIGHT
            };
        }

        // Clear drawing when displaying new layers (after segmentation)
        drawingData = [];

        redrawCanvas();

        const layersContainer = document.getElementById('strokeLayersDisplayContainer');
        layersContainer.innerHTML = '';
        addImagesToDiv(images, layersContainer);
    });
}

// Handle layer selection from the layers panel
function addImagesToDiv(images, divElement, needReverse) {
    divElement.innerHTML = ''; // Clear existing layers

    let keys = Object.keys(images);
    keys = keys.reverse();

    keys.forEach(key => {
        const caption = document.createElement('p');
        let layerNum = key;
        if (needReverse) {
            layerNum = (keys.length - key - 1).toString();
        }
        caption.textContent = `Layer ${layerNum}`;
        const img = document.createElement('img');
        img.src = images[key].src;
        img.classList.add('border-2');
        img.classList.add('rounded-md');

        const figure = document.createElement('figure');
        figure.appendChild(img);
        figure.appendChild(caption);
        divElement.appendChild(figure);

        // Layer click selection callback
        img.addEventListener('click', function () {
            selectedLayer = key;
            // check if we are already selected
            if (img.classList.contains('border-red-300')) {
                img.classList.remove('border-red-300');
                selectedLayer = null;
            } else {
                // Highlight the selected layer visually
                document.querySelectorAll('#strokeLayersDisplayContainer img').forEach(img => {
                    img.classList.remove('border-red-300');
                });
                img.classList.add('border-red-300');
            }
        });
    });
}

function loadImages(sources, callback) {
    // var images = {};
    var loadedImages = 0;
    var numImages = 0;
    // get num of sources
    for (var src in sources) {
        numImages++;
    }
    for (var src in sources) {
        images[src] = new Image();
        images[src].onload = function () {
            if (++loadedImages >= numImages) {
                callback(images);
            }
        };
        images[src].src = sources[src];
    }
}

function clearCanvas() {
    // clear the canvas
    context.clearRect(0, 0, canvas.width, canvas.height);
    for (var key in images) {
        delete images[key];
    }
    for (var key in imagePositions) {
        delete imagePositions[key];
    }
    // Clear drawing data
    drawingData = [];
    // clear sketch layers too 
    const layersContainer = document.getElementById('strokeLayersDisplayContainer');
    layersContainer.innerHTML = '';
}

export function convertCanvasToBWImageUrl() {
    var canvas = document.getElementById('myCanvas');
    var context = canvas.getContext('2d');

    // Get the image data from the canvas
    var imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    var data = imageData.data;

    // Process all pixels in one pass
    for (var i = 0; i < data.length; i += 4) {
        if (data[i + 3] < 125) {  // Very transparent pixels become white
            data[i] = 255;     // Red
            data[i + 1] = 255; // Green
            data[i + 2] = 255; // Blue
            data[i + 3] = 255; // Alpha
        } else {
            const dist_to_white = Math.sqrt(Math.pow(data[i] - 255, 2) + Math.pow(data[i + 1] - 255, 2) + Math.pow(data[i + 2] - 255, 2));
            if (dist_to_white > 100) {
                // turn to black
                data[i] = 0;     // Red
                data[i + 1] = 0; // Green
                data[i + 2] = 0; // Blue
            } else {
                // turn to white
                data[i] = 255;     // Red
                data[i + 1] = 255; // Green
                data[i + 2] = 255; // Blue
            }

            data[i + 3] = 255; // Full opacity
        }
    }

    // Put the modified data back on the canvas
    context.putImageData(imageData, 0, 0);

    return canvas.toDataURL('image/png');
}

export function getInpaintInputs() {
    const dataURL = convertCanvasToBWImageUrl();

    var layerData = [];
    Object.keys(imagePositions).forEach(key => {
        var pos = imagePositions[key];
        layerData.push({
            image_src: images[key].src,
            layerId: key,
            x: pos.x,
            y: pos.y,
            width: pos.width,
            height: pos.height
        });
    });

    const inpaintInputs = { image: dataURL, layerData: layerData, name: loadedImageName }
    return inpaintInputs;
}

/** CALLBACKS **/

function fetchDemosCallback() {
    function fetchGetDemoImages(demo) {
        fetch(`/get-images/${demo}`)
            .then(response => response.json())
            .then(imageUrls => {
                console.log('Image URLs:', imageUrls)
                for (var key in images) {
                    delete images[key];
                }
                for (var key in imagePositions) {
                    delete imagePositions[key];
                }
                displayLayers(imageUrls);
                loadedImageName = demo;
                selectedImage = null;
                document.querySelectorAll('#strokeLayersDisplayContainer img').forEach(img => {
                    img.classList.remove('border-red-400');
                });
                // clear inpainting result
                const inpaintContainer = document.getElementById('inpaintResultContainer');
                inpaintContainer.innerHTML = '';
            })
            .catch(error => console.error('Error loading the images:', error));
    }

    function setupDemoCallback(demos) {
        const container = document.getElementById('buttonContainer');
        if (container) {
            demos.forEach(demo => {
                const button = document.createElement('button');
                button.textContent = `${demo}`;
                button.className = 'loadDemo px-3 py-1.5 bg-white border border-gray-200 rounded-md shadow-sm hover:bg-gray-50 text-lg text-gray-700 transition-colors';
                button.setAttribute('data-demo', demo);
                container.appendChild(button);

                button.addEventListener('click', function () {
                    fetchGetDemoImages(demo);
                });
            });
        }
    }

    fetch('/get-demos')
        .then(response => response.json())
        .then(demos => {
            setupDemoCallback(demos);
        });
}

function canvasOnMouseDown(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const mouseX = (event.clientX - rect.left) * scaleX;
    const mouseY = (event.clientY - rect.top) * scaleY;

    if (drawingMode) {
        startDrawing(mouseX, mouseY);
        return;
    }

    // For moving layers, we still need a selected layer
    if (!selectedLayer) {
        // If no layer is selected but we have images, show a helpful message
        if (Object.keys(images).length > 0) {
            alert('Please select a layer to move, or enable drawing mode to draw :)');
        } else {
            alert('Please load an image or enable drawing mode :)');
        }
        return;
    }

    selectedImage = selectedLayer;
    var pos = imagePositions[selectedImage];
    dragOffsetX = mouseX - pos.x;
    dragOffsetY = mouseY - pos.y;
}

function canvasOnMouseMove(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const mouseX = (event.clientX - rect.left) * scaleX;
    const mouseY = (event.clientY - rect.top) * scaleY;

    if (drawingMode && isDrawing) {
        continueDrawing(mouseX, mouseY);
        return;
    }

    if (selectedImage && !drawingMode) {
        imagePositions[selectedImage].x = mouseX - dragOffsetX;
        imagePositions[selectedImage].y = mouseY - dragOffsetY;
        redrawCanvas();
    }
}

function canvasOnMouseUp() {
    if (drawingMode) {
        stopDrawing();
        return;
    }
    selectedImage = null;
}

function imageLoaderCallback(event) {
    const file = event.target.files[0];
    if (!file) {
        console.error('No file selected!');
        return;
    }

    clearCanvas();

    var reader = new FileReader();
    reader.onload = function (event) {
        var img = new Image();
        img.onload = function () {
            images['0'] = img;
            imagePositions['0'] = { x: 0, y: 0, width: CANVAS_WIDTH, height: CANVAS_HEIGHT };
            redrawCanvas();
        };
        img.src = event.target.result;

        console.log('image src:', img.src)
    };
    reader.readAsDataURL(event.target.files[0]);
    console.log("file name:", event.target.files[0].name)
    loadedImageName = event.target.files[0].name.split('.')[0]

    // Create FormData and append the file
    const formData = new FormData();
    formData.append('image', file);

    // Send the file to the backend
    fetch('/upload-image', { // Update the endpoint URL as per your backend route
        method: 'POST',
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            console.log('Image uploaded successfully:', data);
            // Handle the response as needed
        })
        .catch(error => {
            console.error('Error uploading image:', error);
        });
}

function deleteBtnCallback() {
    if (selectedLayer) {
        delete images[selectedLayer];
        delete imagePositions[selectedLayer];
        // we get negative keys, so we need to reassign the keys
        var newImages = {}
        var newImagePositions = {}
        var counter = 0;
        for (var key in images) {
            newImages[counter.toString()] = images[key];
            newImagePositions[counter.toString()] = imagePositions[key];
            counter++;
        }
        images = newImages;
        imagePositions = newImagePositions;
        redrawCanvas();
        const layersContainer = document.getElementById('strokeLayersDisplayContainer');
        layersContainer.innerHTML = '';
        addImagesToDiv(images, layersContainer);
    }
}

function copyLayerBtnCallback() {
    if (selectedLayer) {
        var newKey = Object.keys(images).length.toString();
        images[newKey] = images[selectedLayer];
        imagePositions[newKey] = { ...imagePositions[selectedLayer] };
        redrawCanvas();
        const layersContainer = document.getElementById('strokeLayersDisplayContainer');
        layersContainer.innerHTML = '';
        addImagesToDiv(images, layersContainer);
    }
}