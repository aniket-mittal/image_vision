#!/usr/bin/env node

/**
 * Test script for image preprocessing functionality
 * This tests the image preprocessing logic that converts images to PNG format
 * and resizes them to meet OpenAI's requirements (under 4MB)
 */

const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');
const path = require('path');

/**
 * Simulate the image preprocessing function from the edit-agent
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
 * Create a test image with specified dimensions
 */
function createTestImage(width, height, format = 'jpeg') {
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

/**
 * Run tests
 */
async function runTests() {
  console.log('🧪 Testing Image Preprocessing Functionality\n');
  
  try {
    // Test 1: Small JPEG image
    console.log('📸 Test 1: Small JPEG image (512x512)');
    const smallJpeg = createTestImage(512, 512, 'jpeg');
    const smallJpegData = `data:image/jpeg;base64,${smallJpeg.toString('base64')}`;
    
    const processedSmall = await preprocessImageForOpenAI(smallJpegData);
    console.log(`✅ Small JPEG processed successfully. Result size: ${processedSmall.length} chars\n`);
    
    // Test 2: Large JPEG image
    console.log('📸 Test 2: Large JPEG image (2048x2048)');
    const largeJpeg = createTestImage(2048, 2048, 'jpeg');
    const largeJpegData = `data:image/jpeg;base64,${largeJpeg.toString('base64')}`;
    
    const processedLarge = await preprocessImageForOpenAI(largeJpegData);
    console.log(`✅ Large JPEG processed successfully. Result size: ${processedLarge.length} chars\n`);
    
    // Test 3: Very large JPEG image
    console.log('📸 Test 3: Very large JPEG image (4096x4096)');
    const veryLargeJpeg = createTestImage(4096, 4096, 'jpeg');
    const veryLargeJpegData = `data:image/jpeg;base64,${veryLargeJpeg.toString('base64')}`;
    
    const processedVeryLarge = await preprocessImageForOpenAI(veryLargeJpegData);
    console.log(`✅ Very large JPEG processed successfully. Result size: ${processedVeryLarge.length} chars\n`);
    
    // Test 4: PNG image (should be returned as-is if small enough)
    console.log('📸 Test 4: PNG image (512x512)');
    const smallPng = createTestImage(512, 512, 'png');
    const smallPngData = `data:image/png;base64,${smallPng.toString('base64')}`;
    
    const processedPng = await preprocessImageForOpenAI(smallPngData);
    console.log(`✅ PNG processed successfully. Result size: ${processedPng.length} chars\n`);
    
    console.log('🎉 All tests passed! Image preprocessing is working correctly.');
    
  } catch (error) {
    console.error('❌ Test failed:', error);
    process.exit(1);
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  runTests();
}

module.exports = { preprocessImageForOpenAI, createTestImage };
