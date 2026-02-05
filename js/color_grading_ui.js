import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ============================================================================
// Color Grading Controller - Live Preview
// ============================================================================

app.registerExtension({
    name: "js-exr-upbitrate.ColorGradingController",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ColorGradingController") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            
            // Store reference image for preview
            this._previewImage = null;
            this._previewCanvas = null;
            this._lastValues = {};
            
            // Create preview widget
            const previewWidget = this.addCustomWidget({
                name: "live_preview",
                type: "preview_canvas",
                value: null,
                draw: (ctx, node, widgetWidth, y, widgetHeight) => {
                    const margin = 10;
                    const previewSize = Math.min(widgetWidth - margin * 2, 150);
                    const x = (widgetWidth - previewSize) / 2;
                    
                    // Background
                    ctx.fillStyle = "#1a1a1a";
                    ctx.strokeStyle = "#333";
                    ctx.lineWidth = 1;
                    ctx.fillRect(x, y, previewSize, previewSize);
                    ctx.strokeRect(x, y, previewSize, previewSize);
                    
                    // Get widget values
                    const getVal = (name, def) => {
                        const w = node.widgets?.find(w => w.name === name);
                        return w ? w.value : def;
                    };
                    
                    const exposure = getVal("exposure", 0);
                    const contrast = getVal("contrast", 1);
                    const lift = getVal("lift", 0);
                    const gamma = getVal("gamma", 1);
                    const gain = getVal("gain", 1);
                    const saturation = getVal("saturation", 1);
                    
                    // Draw gradient preview showing effect
                    const gradientSize = previewSize - 20;
                    const gx = x + 10;
                    const gy = y + 10;
                    
                    // Create gradient that simulates the grading effect
                    for (let i = 0; i < gradientSize; i++) {
                        for (let j = 0; j < gradientSize; j++) {
                            // Base gradient (diagonal)
                            let val = (i + j) / (gradientSize * 2);
                            
                            // Add some color variation
                            let r = val;
                            let g = val * 0.9;
                            let b = val * 0.8;
                            
                            // Apply lift (shadows)
                            r = r + lift;
                            g = g + lift;
                            b = b + lift;
                            
                            // Apply gamma (midtones)
                            r = Math.pow(Math.max(0, r), 1.0 / gamma);
                            g = Math.pow(Math.max(0, g), 1.0 / gamma);
                            b = Math.pow(Math.max(0, b), 1.0 / gamma);
                            
                            // Apply gain (highlights)
                            r = r * gain;
                            g = g * gain;
                            b = b * gain;
                            
                            // Apply exposure
                            const expMult = Math.pow(2, exposure);
                            r = r * expMult;
                            g = g * expMult;
                            b = b * expMult;
                            
                            // Apply contrast
                            const pivot = 0.18;
                            r = pivot * Math.pow(Math.max(0.001, r / pivot), contrast);
                            g = pivot * Math.pow(Math.max(0.001, g / pivot), contrast);
                            b = pivot * Math.pow(Math.max(0.001, b / pivot), contrast);
                            
                            // Apply saturation
                            const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                            r = lum + saturation * (r - lum);
                            g = lum + saturation * (g - lum);
                            b = lum + saturation * (b - lum);
                            
                            // Clamp and convert to 0-255
                            r = Math.min(255, Math.max(0, r * 255));
                            g = Math.min(255, Math.max(0, g * 255));
                            b = Math.min(255, Math.max(0, b * 255));
                            
                            ctx.fillStyle = `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
                            ctx.fillRect(gx + i, gy + j, 1, 1);
                        }
                    }
                    
                    // Draw labels
                    ctx.fillStyle = "#888";
                    ctx.font = "10px monospace";
                    ctx.fillText("Preview", x + 5, y + previewSize + 12);
                    
                    // Show current values
                    ctx.fillStyle = "#666";
                    ctx.font = "9px monospace";
                    const info = `EV:${exposure >= 0 ? '+' : ''}${exposure.toFixed(1)} γ:${gamma.toFixed(1)} S:${saturation.toFixed(1)}`;
                    ctx.fillText(info, x + previewSize - 90, y + previewSize + 12);
                },
                computeSize: function() {
                    return [200, 175];
                }
            });
            
            // Force redraw when any widget changes
            const widgets = ["exposure", "contrast", "lift", "gamma", "gain", "saturation"];
            for (const widget of this.widgets || []) {
                if (widgets.includes(widget.name)) {
                    const origCallback = widget.callback;
                    widget.callback = (value) => {
                        origCallback?.call(widget, value);
                        this.setDirtyCanvas(true, true);
                    };
                }
            }
        };
    }
});


// ============================================================================
// HDR Curve Editor - Interactive curve visualization
// ============================================================================

app.registerExtension({
    name: "js-exr-upbitrate.HDRCurveEditor",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "HDRCurveEditor") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            
            // Add curve preview widget
            const curveWidget = this.addCustomWidget({
                name: "curve_preview",
                type: "curve_display",
                value: null,
                draw: function(ctx, node, widgetWidth, y, widgetHeight) {
                    const margin = 10;
                    const size = Math.min(widgetWidth - margin * 2, 140);
                    const x = (widgetWidth - size) / 2;
                    
                    // Background
                    ctx.fillStyle = "#1a1a1a";
                    ctx.fillRect(x, y, size, size);
                    
                    // Grid
                    ctx.strokeStyle = "#333";
                    ctx.lineWidth = 0.5;
                    for (let i = 0; i <= 4; i++) {
                        const pos = i / 4;
                        ctx.beginPath();
                        ctx.moveTo(x + pos * size, y);
                        ctx.lineTo(x + pos * size, y + size);
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.moveTo(x, y + pos * size);
                        ctx.lineTo(x + size, y + pos * size);
                        ctx.stroke();
                    }
                    
                    // Diagonal reference
                    ctx.strokeStyle = "#444";
                    ctx.lineWidth = 1;
                    ctx.setLineDash([3, 3]);
                    ctx.beginPath();
                    ctx.moveTo(x, y + size);
                    ctx.lineTo(x + size, y);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    
                    // Get curve values
                    const getVal = (name, def) => {
                        const w = node.widgets?.find(w => w.name === name);
                        return w ? w.value : def;
                    };
                    
                    const blacks = getVal("blacks", 0);
                    const shadows = getVal("shadows", 0);
                    const midtones = getVal("midtones", 0);
                    const highlights = getVal("highlights", 0);
                    const whites = getVal("whites", 0);
                    
                    // Build curve points
                    const points = [
                        { x: 0.0, y: Math.max(0, blacks * 0.1) },
                        { x: 0.25, y: 0.25 + shadows * 0.15 },
                        { x: 0.5, y: 0.5 + midtones * 0.2 },
                        { x: 0.75, y: 0.75 + highlights * 0.15 },
                        { x: 1.0, y: Math.min(1, 1.0 + whites * 0.1) },
                    ];
                    
                    // Draw curve
                    ctx.strokeStyle = "#00ff88";
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    
                    for (let i = 0; i <= 100; i++) {
                        const t = i / 100;
                        let lower = 0;
                        for (let j = 0; j < points.length; j++) {
                            if (points[j].x <= t) lower = j;
                        }
                        const upper = Math.min(lower + 1, points.length - 1);
                        
                        let yVal;
                        if (lower === upper) {
                            yVal = points[lower].y;
                        } else {
                            const ratio = (t - points[lower].x) / (points[upper].x - points[lower].x);
                            const smooth = ratio * ratio * (3 - 2 * ratio);
                            yVal = points[lower].y + smooth * (points[upper].y - points[lower].y);
                        }
                        
                        const px = x + t * size;
                        const py = y + size - Math.min(1, Math.max(0, yVal)) * size;
                        
                        if (i === 0) ctx.moveTo(px, py);
                        else ctx.lineTo(px, py);
                    }
                    ctx.stroke();
                    
                    // Draw control points
                    ctx.fillStyle = "#ffffff";
                    for (const pt of points) {
                        const px = x + pt.x * size;
                        const py = y + size - Math.min(1, Math.max(0, pt.y)) * size;
                        ctx.beginPath();
                        ctx.arc(px, py, 4, 0, Math.PI * 2);
                        ctx.fill();
                    }
                    
                    // Labels
                    ctx.fillStyle = "#666";
                    ctx.font = "9px sans-serif";
                    ctx.fillText("B", x + 5, y + size - 5);
                    ctx.fillText("S", x + size * 0.22, y + size - 5);
                    ctx.fillText("M", x + size * 0.47, y + size - 5);
                    ctx.fillText("H", x + size * 0.72, y + size - 5);
                    ctx.fillText("W", x + size - 12, y + size - 5);
                },
                computeSize: function() {
                    return [200, 160];
                }
            });
        };
    }
});


// ============================================================================
// Color Space Converter - Info display
// ============================================================================

app.registerExtension({
    name: "js-exr-upbitrate.ColorSpaceConverter",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ColorSpaceConverter") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            
            const infoWidget = this.addCustomWidget({
                name: "colorspace_info",
                type: "info_display",
                value: null,
                draw: function(ctx, node, widgetWidth, y, widgetHeight) {
                    const margin = 10;
                    
                    const getVal = (name, def) => {
                        const w = node.widgets?.find(w => w.name === name);
                        return w ? w.value : def;
                    };
                    
                    const inputSpace = getVal("input_space", "sRGB");
                    const outputSpace = getVal("output_space", "Linear");
                    
                    // Background
                    ctx.fillStyle = "#1a1a2e";
                    ctx.fillRect(margin, y, widgetWidth - margin * 2, 35);
                    
                    const inputLabel = inputSpace.replace(" (ComfyUI Default)", "");
                    
                    // Input
                    ctx.fillStyle = "#4cc9f0";
                    ctx.font = "bold 11px sans-serif";
                    ctx.fillText(inputLabel, margin + 10, y + 15);
                    
                    // Arrow
                    ctx.fillStyle = "#888";
                    ctx.font = "16px sans-serif";
                    ctx.fillText("→", margin + 80, y + 16);
                    
                    // Output
                    const isLog = outputSpace.includes("Log") || outputSpace.includes("S-Log") || 
                                  outputSpace.includes("V-Log") || outputSpace.includes("DaVinci");
                    ctx.fillStyle = isLog ? "#f72585" : "#00ff88";
                    ctx.font = "bold 11px sans-serif";
                    ctx.fillText(outputSpace, margin + 100, y + 15);
                    
                    // Type indicator
                    ctx.fillStyle = "#444";
                    ctx.font = "9px sans-serif";
                    ctx.fillText(isLog ? "LOG" : "LINEAR", margin + 10, y + 30);
                },
                computeSize: function() {
                    return [200, 40];
                }
            });
        };
    }
});


// ============================================================================
// Save Image EXR - Format indicator
// ============================================================================

app.registerExtension({
    name: "js-exr-upbitrate.SaveImageEXR",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SaveImageEXR") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            
            const formatWidget = this.addCustomWidget({
                name: "format_indicator",
                type: "format_display",
                value: null,
                draw: function(ctx, node, widgetWidth, y, widgetHeight) {
                    const margin = 10;
                    
                    const getVal = (name, def) => {
                        const w = node.widgets?.find(w => w.name === name);
                        return w ? w.value : def;
                    };
                    
                    const bitDepth = getVal("bit_depth", "32");
                    const outputFormat = getVal("output_format", "Linear");
                    const compression = getVal("compression", "zip");
                    
                    // Background
                    ctx.fillStyle = "#0d1b2a";
                    ctx.fillRect(margin, y, widgetWidth - margin * 2, 30);
                    
                    // Bit depth
                    ctx.fillStyle = bitDepth === "32" ? "#00ff88" : "#ffaa00";
                    ctx.font = "bold 12px monospace";
                    ctx.fillText(`${bitDepth}-bit`, margin + 10, y + 20);
                    
                    // Format
                    const isLog = outputFormat !== "Linear";
                    ctx.fillStyle = isLog ? "#f72585" : "#4cc9f0";
                    ctx.font = "11px sans-serif";
                    ctx.fillText(outputFormat, margin + 70, y + 20);
                    
                    // Compression
                    ctx.fillStyle = "#666";
                    ctx.font = "9px sans-serif";
                    ctx.fillText(compression.toUpperCase(), widgetWidth - 45, y + 20);
                },
                computeSize: function() {
                    return [200, 35];
                }
            });
        };
    }
});


// ============================================================================
// Color Match to Reference - Status display
// ============================================================================

app.registerExtension({
    name: "js-exr-upbitrate.ColorMatchToReference",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ColorMatchToReference") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            
            const statusWidget = this.addCustomWidget({
                name: "match_status",
                type: "status_display",
                value: null,
                draw: function(ctx, node, widgetWidth, y, widgetHeight) {
                    const margin = 10;
                    
                    const getVal = (name, def) => {
                        const w = node.widgets?.find(w => w.name === name);
                        return w ? w.value : def;
                    };
                    
                    const matchLum = getVal("match_luminance", true);
                    const matchContrast = getVal("match_contrast", true);
                    const matchColors = getVal("match_colors", true);
                    const strength = getVal("strength", 1.0);
                    
                    // Background
                    ctx.fillStyle = "#1a1a2e";
                    ctx.fillRect(margin, y, widgetWidth - margin * 2, 25);
                    
                    // Status indicators
                    let xPos = margin + 10;
                    ctx.font = "10px sans-serif";
                    
                    // Luminance
                    ctx.fillStyle = matchLum ? "#00ff88" : "#666";
                    ctx.fillText("LUM", xPos, y + 16);
                    xPos += 35;
                    
                    // Contrast
                    ctx.fillStyle = matchContrast ? "#00ff88" : "#666";
                    ctx.fillText("CTR", xPos, y + 16);
                    xPos += 35;
                    
                    // Colors
                    ctx.fillStyle = matchColors ? "#00ff88" : "#666";
                    ctx.fillText("COL", xPos, y + 16);
                    xPos += 40;
                    
                    // Strength
                    ctx.fillStyle = "#888";
                    ctx.fillText(`${Math.round(strength * 100)}%`, xPos, y + 16);
                },
                computeSize: function() {
                    return [200, 30];
                }
            });
        };
    }
});


// ============================================================================
// Advanced Color Match - Algorithm indicator
// ============================================================================

app.registerExtension({
    name: "js-exr-upbitrate.AdvancedColorMatch",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "AdvancedColorMatch") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            
            const methodWidget = this.addCustomWidget({
                name: "method_display",
                type: "method_info",
                value: null,
                draw: function(ctx, node, widgetWidth, y, widgetHeight) {
                    const margin = 10;
                    
                    const getVal = (name, def) => {
                        const w = node.widgets?.find(w => w.name === name);
                        return w ? w.value : def;
                    };
                    
                    const method = getVal("method", "Histogram Matching");
                    const strength = getVal("strength", 1.0);
                    const lumOnly = getVal("match_luminance_only", false);
                    
                    // Background
                    ctx.fillStyle = "#1a2a1a";
                    ctx.fillRect(margin, y, widgetWidth - margin * 2, 50);
                    
                    // Method name with icon
                    const methodColors = {
                        "Histogram Matching": "#00ff88",
                        "LAB Color Space": "#ff8800",
                        "Reinhard Transfer": "#ff00ff",
                        "CLAHE + Histogram": "#00ffff",
                        "CDF Matching": "#ffff00"
                    };
                    
                    const methodIcons = {
                        "Histogram Matching": "📊",
                        "LAB Color Space": "🎨",
                        "Reinhard Transfer": "🔄",
                        "CLAHE + Histogram": "⚡",
                        "CDF Matching": "📈"
                    };
                    
                    ctx.font = "14px sans-serif";
                    ctx.fillText(methodIcons[method] || "🎯", margin + 5, y + 20);
                    
                    ctx.fillStyle = methodColors[method] || "#fff";
                    ctx.font = "bold 12px sans-serif";
                    ctx.fillText(method, margin + 25, y + 20);
                    
                    // Strength bar
                    const barWidth = widgetWidth - margin * 2 - 20;
                    const barHeight = 8;
                    const barY = y + 32;
                    
                    // Background bar
                    ctx.fillStyle = "#333";
                    ctx.fillRect(margin + 10, barY, barWidth, barHeight);
                    
                    // Filled bar
                    ctx.fillStyle = methodColors[method] || "#00ff88";
                    ctx.fillRect(margin + 10, barY, barWidth * strength, barHeight);
                    
                    // Strength text
                    ctx.fillStyle = "#888";
                    ctx.font = "10px sans-serif";
                    ctx.fillText(`${Math.round(strength * 100)}%`, margin + barWidth - 25, barY + 18);
                    
                    // Luminance only indicator
                    if (lumOnly) {
                        ctx.fillStyle = "#ffaa00";
                        ctx.font = "9px sans-serif";
                        ctx.fillText("LUM ONLY", margin + 10, barY + 18);
                    }
                },
                computeSize: function() {
                    return [200, 55];
                }
            });
        };
    }
});


console.log("[js-exr-upbitrate] Color grading UI with live preview loaded");
