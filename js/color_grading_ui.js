import { app } from "../../scripts/app.js";

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
                    const size = Math.min(widgetWidth - margin * 2, 120);
                    const x = margin;
                    
                    // Background
                    ctx.fillStyle = "#1a1a1a";
                    ctx.fillRect(x, y, size, size);
                    
                    // Grid
                    ctx.strokeStyle = "#333";
                    ctx.lineWidth = 0.5;
                    for (let i = 0; i <= 4; i++) {
                        const pos = i / 4;
                        // Vertical
                        ctx.beginPath();
                        ctx.moveTo(x + pos * size, y);
                        ctx.lineTo(x + pos * size, y + size);
                        ctx.stroke();
                        // Horizontal
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
                    
                    // Get curve values from widgets
                    const getWidgetValue = (name, defaultVal) => {
                        const w = node.widgets?.find(w => w.name === name);
                        return w ? w.value : defaultVal;
                    };
                    
                    const blacks = getWidgetValue("blacks", 0);
                    const shadows = getWidgetValue("shadows", 0);
                    const midtones = getWidgetValue("midtones", 0);
                    const highlights = getWidgetValue("highlights", 0);
                    const whites = getWidgetValue("whites", 0);
                    
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
                        
                        // Find surrounding points and interpolate
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
                        const py = y + size - yVal * size;
                        
                        if (i === 0) ctx.moveTo(px, py);
                        else ctx.lineTo(px, py);
                    }
                    ctx.stroke();
                    
                    // Draw control points
                    ctx.fillStyle = "#ffffff";
                    for (const pt of points) {
                        const px = x + pt.x * size;
                        const py = y + size - pt.y * size;
                        ctx.beginPath();
                        ctx.arc(px, py, 4, 0, Math.PI * 2);
                        ctx.fill();
                    }
                    
                    // Labels
                    ctx.fillStyle = "#888";
                    ctx.font = "10px monospace";
                    ctx.fillText("S", x + size * 0.2, y + size - 5);
                    ctx.fillText("M", x + size * 0.45, y + size - 5);
                    ctx.fillText("H", x + size * 0.7, y + size - 5);
                },
                computeSize: function() {
                    return [150, 130];
                }
            });
            
            // Force redraw when values change
            const origOnChange = this.onPropertyChanged;
            this.onPropertyChanged = function(name, value) {
                origOnChange?.apply(this, arguments);
                this.setDirtyCanvas(true, true);
            };
        };
    }
});


// ============================================================================
// Color Grading Controller - Visual preview
// ============================================================================

app.registerExtension({
    name: "js-exr-upbitrate.ColorGradingController",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ColorGradingController") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            
            // Add visual indicator widget
            const indicator = this.addCustomWidget({
                name: "grading_indicator",
                type: "grading_viz",
                value: null,
                draw: function(ctx, node, widgetWidth, y, widgetHeight) {
                    const margin = 10;
                    const barHeight = 20;
                    const barWidth = widgetWidth - margin * 2;
                    
                    // Get values
                    const getWidgetValue = (name, defaultVal) => {
                        const w = node.widgets?.find(w => w.name === name);
                        return w ? w.value : defaultVal;
                    };
                    
                    const exposure = getWidgetValue("exposure", 0);
                    const contrast = getWidgetValue("contrast", 1);
                    const lift = getWidgetValue("lift", 0);
                    const gamma = getWidgetValue("gamma", 1);
                    const gain = getWidgetValue("gain", 1);
                    const saturation = getWidgetValue("saturation", 1);
                    
                    // Draw gradient bar (simulating effect)
                    const gradient = ctx.createLinearGradient(margin, y, margin + barWidth, y);
                    
                    // Simulate lift/gamma/gain effect on gradient
                    for (let i = 0; i <= 10; i++) {
                        const t = i / 10;
                        let val = t;
                        
                        // Apply lift
                        val = val + lift;
                        // Apply gamma
                        val = Math.pow(Math.max(0, val), 1.0 / gamma);
                        // Apply gain
                        val = val * gain;
                        // Apply exposure
                        val = val * Math.pow(2, exposure);
                        // Apply contrast
                        val = 0.18 * Math.pow(Math.max(0.001, val / 0.18), contrast);
                        
                        val = Math.min(1, Math.max(0, val));
                        
                        // Saturation affects color vibrancy
                        const sat = Math.min(1, saturation);
                        const r = val;
                        const g = val * (1 - (1 - sat) * 0.3);
                        const b = val * (1 - (1 - sat) * 0.6);
                        
                        gradient.addColorStop(t, `rgb(${Math.round(r*255)}, ${Math.round(g*255)}, ${Math.round(b*255)})`);
                    }
                    
                    ctx.fillStyle = gradient;
                    ctx.fillRect(margin, y, barWidth, barHeight);
                    
                    // Border
                    ctx.strokeStyle = "#555";
                    ctx.strokeRect(margin, y, barWidth, barHeight);
                    
                    // Labels
                    ctx.fillStyle = "#aaa";
                    ctx.font = "9px sans-serif";
                    ctx.fillText("Shadows", margin, y + barHeight + 12);
                    ctx.fillText("Highlights", margin + barWidth - 45, y + barHeight + 12);
                    
                    // Show values
                    ctx.fillStyle = "#666";
                    ctx.font = "10px monospace";
                    const info = `EV:${exposure > 0 ? '+' : ''}${exposure.toFixed(1)} C:${contrast.toFixed(2)} S:${saturation.toFixed(1)}`;
                    ctx.fillText(info, margin, y + barHeight + 25);
                },
                computeSize: function() {
                    return [200, 45];
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
            
            // Add color space info widget
            const infoWidget = this.addCustomWidget({
                name: "colorspace_info",
                type: "info_display",
                value: null,
                draw: function(ctx, node, widgetWidth, y, widgetHeight) {
                    const margin = 10;
                    
                    const getWidgetValue = (name, defaultVal) => {
                        const w = node.widgets?.find(w => w.name === name);
                        return w ? w.value : defaultVal;
                    };
                    
                    const inputSpace = getWidgetValue("input_space", "sRGB");
                    const outputSpace = getWidgetValue("output_space", "Linear");
                    
                    // Draw conversion arrow
                    ctx.fillStyle = "#1a1a2e";
                    ctx.fillRect(margin, y, widgetWidth - margin * 2, 30);
                    
                    ctx.fillStyle = "#4cc9f0";
                    ctx.font = "11px sans-serif";
                    
                    const inputLabel = inputSpace.replace(" (ComfyUI Default)", "");
                    const arrowX = margin + 60;
                    
                    ctx.fillText(inputLabel, margin + 5, y + 12);
                    
                    // Arrow
                    ctx.fillStyle = "#888";
                    ctx.fillText("→", arrowX + 20, y + 12);
                    
                    ctx.fillStyle = "#f72585";
                    ctx.fillText(outputSpace, arrowX + 40, y + 12);
                    
                    // Log indicator
                    if (outputSpace.includes("Log") || outputSpace.includes("S-Log") || 
                        outputSpace.includes("V-Log") || outputSpace.includes("DaVinci")) {
                        ctx.fillStyle = "#00ff88";
                        ctx.font = "9px sans-serif";
                        ctx.fillText("LOG", widgetWidth - 35, y + 25);
                    }
                },
                computeSize: function() {
                    return [200, 35];
                }
            });
        };
    }
});


// ============================================================================
// Image Stats - Statistics display
// ============================================================================

app.registerExtension({
    name: "js-exr-upbitrate.ImageStats",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ImageStats") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            
            // The stats output will show in the connected text widget
            // Add a header
            this.addWidget("text", "info", "Connect to show image statistics", () => {});
        };
    }
});


// ============================================================================
// Save Image EXR - Log format indicator
// ============================================================================

app.registerExtension({
    name: "js-exr-upbitrate.SaveImageEXR",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SaveImageEXR") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            
            // Add format indicator
            const formatWidget = this.addCustomWidget({
                name: "format_indicator",
                type: "format_display",
                value: null,
                draw: function(ctx, node, widgetWidth, y, widgetHeight) {
                    const margin = 10;
                    
                    const getWidgetValue = (name, defaultVal) => {
                        const w = node.widgets?.find(w => w.name === name);
                        return w ? w.value : defaultVal;
                    };
                    
                    const bitDepth = getWidgetValue("bit_depth", "32");
                    const outputFormat = getWidgetValue("output_format", "Linear");
                    const compression = getWidgetValue("compression", "zip");
                    
                    // Background
                    ctx.fillStyle = "#0d1b2a";
                    ctx.fillRect(margin, y, widgetWidth - margin * 2, 25);
                    
                    // Bit depth badge
                    ctx.fillStyle = bitDepth === "32" ? "#00ff88" : "#ffaa00";
                    ctx.font = "bold 11px monospace";
                    ctx.fillText(`${bitDepth}-bit`, margin + 5, y + 16);
                    
                    // Format badge
                    const isLog = outputFormat !== "Linear";
                    ctx.fillStyle = isLog ? "#f72585" : "#4cc9f0";
                    ctx.font = "10px sans-serif";
                    ctx.fillText(outputFormat, margin + 55, y + 16);
                    
                    // Compression
                    ctx.fillStyle = "#666";
                    ctx.font = "9px sans-serif";
                    ctx.fillText(compression.toUpperCase(), widgetWidth - 40, y + 16);
                },
                computeSize: function() {
                    return [200, 30];
                }
            });
        };
    }
});

console.log("[js-exr-upbitrate] Color grading UI extensions loaded");
