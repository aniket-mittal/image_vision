#!/usr/bin/env node

/**
 * Test script for OpenAI integration
 * This tests the OpenAI API calls and helps debug any issues
 */

const fs = require('fs');
const path = require('path');

// Simulate the environment variables
process.env.OPENAI_API_KEY = process.env.OPENAI_API_KEY || 'sk-test-key';

// Simulate the canvas library
const { createCanvas, loadImage } = require('canvas');

/**
 * Simulate the image preprocessing function
 */
async function preprocessImageForOpenAI(imageData, maxSizeBytes = 4 * 1024 * 1024) {
  try {
    console.log(`Preprocessing image: original size ${imageData.length} chars`);
    
    // Remove data URL prefix to get base64 data
    const base64Data = imageData.replace(/^data:image\/[a-z]+;base64,/, '');
    const imageBuffer = Buffer.from(base64Data, 'base64');
    
    console.log(`Image buffer size: ${imageBuffer.length} bytes`);
    
    // Check if image is already small enough
    if (imageBuffer.length <= maxSizeBytes) {
      // If it's already PNG and small enough, return as is
      if (imageData.includes('image/png')) {
        console.log(`Image already PNG and under size limit, returning as-is`);
        return imageData;
      }
    }
    
    // Load the image using canvas
    const img = await loadImage(imageBuffer);
    console.log(`Original image dimensions: ${img.width}x${img.height}`);
    
    // Calculate target dimensions to reduce file size
    let targetWidth = img.width;
    let targetHeight = img.height;
    
    // If image is very large, scale it down proportionally
    if (img.width > 1024 || img.height > 1024) {
      const aspectRatio = img.width / img.height;
      if (aspectRatio > 1) {
        targetWidth = 1024;
        targetHeight = Math.round(1024 / aspectRatio);
      } else {
        targetHeight = 1024;
        targetWidth = Math.round(1024 * aspectRatio);
      }
      console.log(`Resizing to: ${targetWidth}x${targetHeight}`);
    }
    
    // Create canvas with target dimensions
    const canvas = createCanvas(targetWidth, targetHeight);
    const ctx = canvas.getContext('2d');
    
    // Draw the resized image
    ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
    
    // Convert to PNG with high quality but reasonable compression
    const pngBuffer = canvas.toBuffer('image/png', { 
      compressionLevel: 6 // Balance between quality and size
    });
    
    console.log(`Initial PNG size: ${pngBuffer.length} bytes`);
    
    // Check if the PNG is still too large
    if (pngBuffer.length > maxSizeBytes) {
      console.log(`Image still too large, applying aggressive compression`);
      // Further reduce quality by scaling down more
      const scaleFactor = Math.sqrt(maxSizeBytes / pngBuffer.length) * 0.9; // 10% buffer
      targetWidth = Math.round(targetWidth * scaleFactor);
      targetHeight = Math.round(targetHeight * scaleFactor);
      
      console.log(`Further resizing to: ${targetWidth}x${targetHeight}`);
      
      const smallCanvas = createCanvas(targetWidth, targetHeight);
      const smallCtx = smallCanvas.getContext('2d');
      smallCtx.drawImage(img, 0, 0, targetWidth, targetHeight);
      
      const finalBuffer = smallCanvas.toBuffer('image/png', { 
        compressionLevel: 9 // Maximum compression
      });
      
      console.log(`Final processed image size: ${finalBuffer.length} bytes`);
      return `data:image/png;base64,${finalBuffer.toString('base64')}`;
    }
    
    console.log(`Final processed image size: ${pngBuffer.length} bytes`);
    return `data:image/png;base64,${pngBuffer.toString('base64')}`;
  } catch (error) {
    console.error('Image preprocessing failed:', error);
    throw error;
  }
}

/**
 * Create a test mask (white circle on black background)
 */
function createTestMask(width, height) {
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  
  // Fill with black background
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, width, height);
  
  // Draw white circle in center (this represents the area to edit)
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(width/2, height/2, Math.min(width, height)/4, 0, 2 * Math.PI);
  ctx.fill();
  
  return canvas.toBuffer('image/png');
}

/**
 * Test OpenAI API integration
 */
async function testOpenAIIntegration() {
  console.log('🧪 Testing OpenAI Integration\n');
  
  try {
    // Check API key
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey || apiKey === 'sk-test-key') {
      console.log('⚠️  No valid OPENAI_API_KEY found. Set it as an environment variable to test actual API calls.');
      console.log('   Example: OPENAI_API_KEY=sk-your-key-here node test_openai_integration.js\n');
      return;
    }
    
    console.log(`✅ API key found: ${apiKey.substring(0, 7)}...`);
    
    // Create test image and mask
    console.log('\n📸 Creating test image and mask...');
    const testImage = createTestImage(512, 512, 'png');
    const testMask = createTestMask(512, 512);
    
    const imageData = `data:image/png;base64,${testImage.toString('base64')}`;
    const maskData = `data:image/png;base64,${testMask.toString('base64')}`;
    
    console.log(`✅ Test image created: ${testImage.length} bytes`);
    console.log(`✅ Test mask created: ${testMask.length} bytes`);
    
    // Test preprocessing
    console.log('\n🔧 Testing image preprocessing...');
    const processedImage = await preprocessImageForOpenAI(imageData);
    const processedMask = await preprocessImageForOpenAI(maskData);
    
    console.log(`✅ Image preprocessing successful`);
    console.log(`✅ Mask preprocessing successful`);
    
    // Test FormData creation
    console.log('\n📤 Testing FormData creation...');
    const { FormData } = require('formdata-node');
    
    const form = new FormData();
    form.append("image", new Blob([Buffer.from(processedImage.split(",")[1], "base64")], { type: "image/png" }), "image.png");
    form.append("mask", new Blob([Buffer.from(processedMask.split(",")[1], "base64")], { type: "image/png" }), "mask.png");
    form.append("prompt", "Remove the object in the center");
    form.append("size", "1024x1024");
    
    console.log(`✅ FormData created successfully`);
    console.log(`   - Image: ${Buffer.from(processedImage.split(",")[1], "base64").length} bytes`);
    console.log(`   - Mask: ${Buffer.from(processedMask.split(",")[1], "base64").length} bytes`);
    console.log(`   - Prompt: "Remove the object in the center"`);
    console.log(`   - Size: 1024x1024`);
    
    // Test actual API call (if API key is valid)
    console.log('\n🚀 Testing OpenAI API call...');
    
    const fetch = require('node-fetch');
    const response = await fetch("https://api.openai.com/v1/images/edits", {
      method: "POST",
      headers: { 
        Authorization: `Bearer ${apiKey}`,
        // Note: Don't set Content-Type for FormData, let the browser set it
      },
      body: form,
    });
    
    console.log(`✅ API response received: ${response.status} ${response.statusText}`);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ API error: ${errorText}`);
      return;
    }
    
    const data = await response.json();
    console.log(`✅ API response parsed successfully`);
    console.log(`   Response keys: ${Object.keys(data)}`);
    
    if (data.data && data.data[0]) {
      console.log(`   Data[0] keys: ${Object.keys(data.data[0])}`);
      
      if (data.data[0].b64_json) {
        console.log(`✅ Found b64_json field (${data.data[0].b64_json.length} chars)`);
      } else if (data.data[0].url) {
        console.log(`✅ Found URL field: ${data.data[0].url}`);
      } else {
        console.log(`⚠️  No b64_json or URL found in response`);
      }
    }
    
    console.log('\n🎉 OpenAI integration test completed successfully!');
    
  } catch (error) {
    console.error('❌ Test failed:', error);
    console.error('Stack trace:', error.stack);
  }
}

/**
 * Create a test image with specified dimensions
 */
function createTestImage(width, height, format = 'png') {
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  
  // Create a gradient background
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, '#ff0000');
  gradient.addColorStop(0.5, '#00ff00');
  gradient.addColorStop(1, '#0000ff');
  
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  
  // Add some shapes
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(width/4, height/4, width/2, height/2);
  
  ctx.fillStyle = '#000000';
  ctx.beginPath();
  ctx.arc(width/2, height/2, Math.min(width, height)/8, 0, 2 * Math.PI);
  ctx.fill();
  
  // Convert to specified format
  if (format === 'jpeg') {
    return canvas.toBuffer('image/jpeg', { quality: 0.9 });
  } else {
    return canvas.toBuffer('image/png');
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  testOpenAIIntegration();
}

module.exports = { 
  preprocessImageForOpenAI, 
  createTestImage, 
  createTestMask,
  testOpenAIIntegration 
};
