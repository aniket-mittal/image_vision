// Simple test for crop and lasso API functionality
const fs = require('fs');
const path = require('path');

// Test data
const testData = {
  imageData: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=", // Minimal base64 image
  userQuery: "What can you see in this image?",
  toolType: "crop",
  coordinates: [100, 100, 200, 200]
};

async function testCropLassoAPI() {
  try {
    console.log('Testing crop-lasso API...');
    
    const response = await fetch('http://localhost:3000/api/crop-lasso-process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testData)
    });

    if (response.ok) {
      const result = await response.json();
      console.log('✅ API test successful!');
      console.log('Processing type:', result.processingType);
      console.log('Answer length:', result.answer?.length || 0);
      console.log('Has processed image:', !!result.processedImageData);
    } else {
      const errorText = await response.text();
      console.error('❌ API test failed:', response.status, errorText);
    }
  } catch (error) {
    console.error('❌ Test error:', error.message);
  }
}

// Run test if this file is executed directly
if (require.main === module) {
  testCropLassoAPI();
}

module.exports = { testCropLassoAPI }; 