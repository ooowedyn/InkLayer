/***************************************************************************************************
 *
 * -- Adobe Illustrator Script for Brush Augmentation --
 *
 * Version: 2.0
 * Author: Chuan Yan (chuanyan@stanford.edu)
 * Date: August 10, 2025
 *
 * Description:
 * This script automates the process of applying various artistic brush styles to a batch of
 * SVG files. It then exports the stylized images as PNG files. It's designed for generating
 * training datasets or benchmark images by applying consistent or randomized brush effects.
 *
 * Copyright Chuan Yan (chuanyan@stanford.edu) 2024, All rights reserved.
 * This software is provided for non-commercial use only. Any commercial use, reproduction,
 * or distribution without prior written permission is strictly prohibited.
 *
 ***************************************************************************************************
 *
 * !!! HOW TO USE - PLEASE READ CAREFULLY !!!
 *
 * 1. CONFIGURE PARAMETERS:
 * - Scroll down to the "SCRIPT_CONFIG" section below.
 * - Carefully edit the settings to match your project needs. This is the most important step.
 *
 * 2. FOLDER STRUCTURE:
 * - This script runs from its current location. All paths in the config are relative to it.
 * - Create an "svg" folder next to this script and place your input SVG files inside.
 * (e.g., /your_project_folder/svg/0001.svg)
 * - Create a "png" folder next to this script. This is where the output will be saved.
 * (e.g., /your_project_folder/png/)
 * - Inside the "png" folder, create subfolders for each brush you want to use. The folder
 * name must be a two-digit number corresponding to the brush's index in the config.
 * For example, if you set SELECTED_BRUSH to 1, you MUST create a folder named "01".
 * (e.g., /your_project_folder/png/01/)
 *
 * 3. BRUSH TEMPLATE:
 * - Make sure your brush template file (e.g., "brush02.ai") is in the same folder as this script.
 * - The script will open this file to load the custom brushes into Illustrator.
 *
 * 4. RUN THE SCRIPT IN ILLUSTRATOR:
 * - Open Adobe Illustrator.
 * - Go to File > Scripts > Other Script... (or press Ctrl+F12 / Cmd+F12).
 * - Navigate to this script file (`brush_augmentation.jsx`) and open it.
 * - The script will start processing the files automatically.
 *
 * 5. MEMORY MANAGEMENT:
 * - Illustrator can consume a lot of RAM during this process. The script is designed to
 * process a batch of files (`MAX_PROCESSED_FILES`) and then stop.
 * - After the script finishes a batch, it's highly recommended to RESTART Illustrator
 * to release memory before running the script on the next batch.
 *
 ***************************************************************************************************/


/**
 * @section SCRIPT_CONFIG
 * @description All user-configurable parameters are here. Edit these before running.
 */
var SCRIPT_CONFIG = {
    // --- File Path Settings ---
    // All paths are relative to the location of this script file.
    BRUSH_TEMPLATE_FILE: "brush02.ai",      // The Adobe Illustrator file containing your custom brushes.
    SVG_INPUT_FOLDER: "svg",                // Folder containing the source SVG files (e.g., 0001.svg, 0002.svg).
    PNG_OUTPUT_FOLDER: "png",               // Root folder where the final PNGs will be saved.

    // --- Batch Processing Settings ---
    FILE_BATCH_SIZE: 5,                     // Number of SVG files to process in a single run.
    MAX_PROCESSED_FILES: 1000,              // The script will stop after processing this many files.
                                            // IMPORTANT: Restart Illustrator after each batch to clear memory.

    // --- Brush Definitions ---
    BRUSHES: {
        // This is a list of brush names as they appear in your BRUSH_TEMPLATE_FILE.
        // The script uses the index of the brush in this list to identify it.
        // Index 0 is reserved for the default "Basic" stroke. DO NOT REMOVE `null`.
        NAMES: [
            null,                           // Index 0: Basic Stroke (no brush applied)
            "Calligraphic Brush 1",         // Index 1
            "BrushPen 111",                 // Index 2
            "HEJ_BLACK_STROKE_01",          // Index 3
            "HEJ_TRUE_GRIS_M_STROKE_01",    // Index 4
            "Lino Cut 8",                   // Index 5
            "BrushPen 42",                  // Index 6
            "Charcoal_smudged_3",           // Index 7
            "HEJ_ANGRY_STROKE_03",          // Index 8
            'HEJ_TRUE_GRIS_M_STROKE_04',    // Index 9
            'Graphite - B_short',           // Index 10
            "Comic Book_Contrast 3"         // Index 11
        ],
        // Set this to the index of the brush you want to apply from the NAMES list above.
        // REMEMBER: Create a corresponding two-digit folder in your PNG_OUTPUT_FOLDER.
        // (e.g., for brush 1, create a folder named "01").
        SELECTED_BRUSH_INDEX: 9
    },

    // --- PNG Export Options ---
    // These settings control the quality and format of the output PNG files.
    // It's usually not necessary to change these.
    EXPORT_OPTIONS_PNG: {
        antiAliasing: true,
        transparency: true,
        artBoardClipping: true,
        horizontalScale: 100,
        verticalScale: 100,
        resolution: 72, // DPI
        matte: false
    }
};


/**
 * Main entry point of the script.
 */
function main() {
    var config = initializeConfig();
    var processedFileCount = 0;
    var searchStartIndex = 0;

    while (processedFileCount < SCRIPT_CONFIG.MAX_PROCESSED_FILES) {
        var svgFiles = findSvgFiles(config.svgPath, searchStartIndex, SCRIPT_CONFIG.FILE_BATCH_SIZE);
        if (svgFiles.length === 0) {
            alert("Processing complete. No more SVG files found to process.");
            break;
        }

        var openedDocs = [];
        var brushesLoaded = false;

        for (var i = 0; i < svgFiles.length; i++) {
            var svgFile = svgFiles[i];
            var baseName = svgFile.name.split('.')[0];
            var pngFile = new File(config.pngOutputPath + "/" + baseName + ".png");
            var augmentedSvgFile = new File(config.pngOutputPath + "/" + baseName + ".svg");

            // Skip if the output file already exists
            if (augmentedSvgFile.exists) {
                continue;
            }

            // Lazily load brushes only when the first valid file is found
            if (!brushesLoaded) {
                if (!loadBrushesFromTemplate(config.brushTemplateFile)) {
                    return; // Stop script if brushes can't be loaded
                }
                brushesLoaded = true;
            }

            var doc = app.open(svgFile);
            openedDocs.push(doc);

            applyBrushAndExport(doc, config, pngFile, augmentedSvgFile);
            processedFileCount++;

            if (processedFileCount >= SCRIPT_CONFIG.MAX_PROCESSED_FILES) {
                break; // Stop if max file limit is reached
            }
        }

        // Close all documents opened in this batch
        closeDocuments(openedDocs);
        
        // Update the search index for the next batch
        searchStartIndex += SCRIPT_CONFIG.FILE_BATCH_SIZE;
    }

    if (processedFileCount >= SCRIPT_CONFIG.MAX_PROCESSED_FILES) {
        alert("Maximum file limit (" + SCRIPT_CONFIG.MAX_PROCESSED_FILES + ") reached.\nPlease restart Illustrator to free up memory before running again.");
    }
}

/**
 * Initializes and resolves all paths and options from the global config.
 * @returns {Object} An object containing resolved paths and settings.
 */
function initializeConfig() {
    var scriptPath = new File($.fileName).parent;
    var brushIndex = SCRIPT_CONFIG.BRUSHES.SELECTED_BRUSH_INDEX;
    var brushSubfolder = ("00" + brushIndex).slice(-2);

    var config = {
        brushTemplateFile: new File(scriptPath + '/' + SCRIPT_CONFIG.BRUSH_TEMPLATE_FILE),
        svgPath: scriptPath + '/' + SCRIPT_CONFIG.SVG_INPUT_FOLDER,
        pngOutputPath: scriptPath + '/' + SCRIPT_CONFIG.PNG_OUTPUT_FOLDER + '/' + brushSubfolder,
        selectedBrushName: SCRIPT_CONFIG.BRUSHES.NAMES[brushIndex],
        exportOptions: new ImageCaptureOptions()
    };
    
    // Populate export options
    var pngOpts = SCRIPT_CONFIG.EXPORT_OPTIONS_PNG;
    config.exportOptions.artBoardClipping = pngOpts.artBoardClipping;
    config.exportOptions.resolution = pngOpts.resolution;
    config.exportOptions.antiAliasing = pngOpts.antiAliasing;
    config.exportOptions.matte = pngOpts.matte;
    config.exportOptions.horizontalScale = pngOpts.horizontalScale;
    config.exportOptions.verticalScale = pngOpts.verticalScale;
    config.exportOptions.transparency = pngOpts.transparency;

    return config;
}

/**
 * Searches for and returns a batch of SVG files.
 * @param {String} svgFolderPath - The path to the SVG folder.
 * @param {Number} startIndex - The file number to start searching from.
 * @param {Number} batchSize - The number of files to find.
 * @returns {Array} An array of File objects.
 */
function findSvgFiles(svgFolderPath, startIndex, batchSize) {
    var fileList = [];
    for (var i = 0; i < batchSize; i++) {
        var fileIndex = startIndex + i;
        var fileName = ("0000" + fileIndex).slice(-4) + ".svg";
        var svgFile = new File(svgFolderPath + "/" + fileName);
        if (svgFile.exists) {
            fileList.push(svgFile);
        }
    }
    return fileList;
}

/**
 * Loads brushes from the template file by opening it, copying contents, and closing it.
 * @param {File} brushTemplateFile - The brush template .ai file.
 * @returns {Boolean} True if successful, false otherwise.
 */
function loadBrushesFromTemplate(brushTemplateFile) {
    if (SCRIPT_CONFIG.BRUSHES.SELECTED_BRUSH_INDEX === 0) {
        return true; // No brush needed for "Basic" stroke
    }

    if (!brushTemplateFile.exists) {
        alert("Error: Brush template file not found at:\n" + brushTemplateFile.fsName);
        return false;
    }

    // Check if the template is already open to avoid reopening
    for (var i = 0; i < app.documents.length; i++) {
        if (app.documents[i].name === brushTemplateFile.name) {
            app.documents[i].close(SaveOptions.DONOTSAVECHANGES);
        }
    }

    var docBrushes = app.open(brushTemplateFile);
    docBrushes.selectObjectsOnActiveArtboard();
    app.copy();
    docBrushes.close(SaveOptions.DONOTSAVECHANGES);
    return true;
}

/**
 * Applies the selected brush to all paths in the document and exports it.
 * @param {Document} doc - The active document to process.
 * @param {Object} config - The initialized configuration object.
 * @param {File} pngFile - The destination file for the PNG export.
 * @param {File} svgFile - The destination file for the augmented SVG.
 */
function applyBrushAndExport(doc, config, pngFile, svgFile) {
    app.activeDocument = doc;

    // Paste the brush definitions into the current document's library
    if (config.selectedBrushName) {
        app.executeMenuCommand("paste");
        app.executeMenuCommand("clear"); // Deselect pasted items
    }

    var brush = findBrushInDocument(doc, config.selectedBrushName);
    if (!brush && config.selectedBrushName) {
        alert("Warning: Brush '" + config.selectedBrushName + "' not found in the active document.");
        return;
    }

    // Apply the brush to all path items
    for (var i = 0; i < doc.pathItems.length; i++) {
        var item = doc.pathItems[i];
        if (item.stroked && !isFilledWithWhite(item)) {
            if (brush) {
                item.strokeBrush = brush;
                // Optional: Adjust stroke width dynamically here if needed
                // item.strokeWidth = ...;
            } else {
                // Handle basic stroke case (no brush)
                item.strokeWidth = 0.1;
            }
        }
    }

    // Export the final results
    try {
        saveAsSVG(svgFile);
        var activeArtboard = doc.artboards[doc.artboards.getActiveArtboardIndex()];
        doc.imageCapture(pngFile, activeArtboard.artboardRect, config.exportOptions);
    } catch (e) {
        alert("Error exporting file: " + doc.name + "\n" + e);
    }
}

/**
 * Finds a specific brush by name within the active document.
 * @param {Document} doc - The document to search within.
 * @param {String} brushName - The name of the brush to find.
 * @returns {Brush|null} The found brush object or null.
 */
function findBrushInDocument(doc, brushName) {
    if (!brushName) return null;
    for (var i = 0; i < doc.brushes.length; i++) {
        if (doc.brushes[i].name === brushName) {
            return doc.brushes[i];
        }
    }
    return null;
}

/**
 * Saves the active document as an SVG file.
 * @param {File} outputPath - The file path to save the SVG to.
 */
function saveAsSVG(outputPath) {
    var exportOptions = new ExportOptionsSVG();
    exportOptions.embedRasterImages = true;
    exportOptions.embedAllFonts = false;
    exportOptions.fontSubsetting = SVGFontSubsetting.None;
    app.activeDocument.exportFile(outputPath, ExportType.SVG, exportOptions);
}

/**
 * Checks if a path item is filled with solid white color.
 * @param {PathItem} path - The item to check.
 * @returns {Boolean} True if the fill is white.
 */
function isFilledWithWhite(path) {
    if (path.filled && path.fillColor) {
        var color = path.fillColor;
        if (color.typename === "RGBColor") {
            return color.red === 255 && color.green === 255 && color.blue === 255;
        }
        if (color.typename === "GrayColor") {
            return color.gray === 0; // In scripting, 0 is white for grayscale, 100 is black.
        }
        if (color.typename === "CMYKColor") {
            return color.cyan === 0 && color.magenta === 0 && color.yellow === 0 && color.black === 0;
        }
    }
    return false;
}

/**
 * Closes all documents in the provided list without saving changes.
 * @param {Array<Document>} docs - An array of document objects to close.
 */
function closeDocuments(docs) {
    for (var i = 0; i < docs.length; i++) {
        docs[i].close(SaveOptions.DONOTSAVECHANGES);
    }
}

// --- Script Execution ---
// The script starts running from here.
main();
