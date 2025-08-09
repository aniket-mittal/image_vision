const fs = require('fs');
const path = require('path');

// Read a test image and convert to base64
const testImagePath = path.join(__dirname, 'testing_images/images/v1_0.png');
const imageBuffer = fs.readFileSync(testImagePath);
const base64Image = `data:image/png;base64,${imageBuffer.toString('base64')}`;

async function testAPI() {
  console.log('Testing API integration...');
  
  // Test Auto mode
  try {
    console.log('\n1. Testing Auto mode...');
    const autoResponse = await fetch('http://localhost:3000/api/auto-process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        imageData: base64Image,
        userQuery: 'Where is the cat facing?'
      }),
    });
    
    if (autoResponse.ok) {
      const autoResult = await autoResponse.json();
      console.log('✅ Auto mode successful');
      console.log('Processing type:', autoResult.processingType);
      console.log('Refined query:', autoResult.refinedQuery);
      console.log('Has processed image:', !!autoResult.processedImageData);
    } else {
      console.log('❌ Auto mode failed:', autoResponse.statusText);
    }
  } catch (error) {
    console.log('❌ Auto mode error:', error.message);
  }

  // Test Attention mode
  try {
    console.log('\n2. Testing Attention mode...');
    const attentionResponse = await fetch('http://localhost:3000/api/attention-process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        imageData: base64Image,
        userQuery: 'Find the cat'
      }),
    });
    
    if (attentionResponse.ok) {
      const attentionResult = await attentionResponse.json();
      console.log('✅ Attention mode successful');
      console.log('Refined query:', attentionResult.refinedQuery);
      console.log('Has processed image:', !!attentionResult.processedImageData);
    } else {
      console.log('❌ Attention mode failed:', attentionResponse.statusText);
    }
  } catch (error) {
    console.log('❌ Attention mode error:', error.message);
  }

  // Test Segmentation mode
  try {
    console.log('\n3. Testing Segmentation mode...');
    const segmentationResponse = await fetch('http://localhost:3000/api/segmentation-process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        imageData: base64Image,
        userQuery: 'Segment the cat'
      }),
    });
    
    if (segmentationResponse.ok) {
      const segmentationResult = await segmentationResponse.json();
      console.log('✅ Segmentation mode successful');
      console.log('Refined query:', segmentationResult.refinedQuery);
      console.log('Has processed image:', !!segmentationResult.processedImageData);
    } else {
      console.log('❌ Segmentation mode failed:', segmentationResponse.statusText);
    }
  } catch (error) {
    console.log('❌ Segmentation mode error:', error.message);
  }

  console.log('\n🎉 Integration test completed!');
}

testAPI().catch(console.error); 