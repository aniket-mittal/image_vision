const fs = require('fs');
const path = require('path');

// Read a test image and convert to base64
const testImagePath = path.join(__dirname, 'testing_images/images/v1_0.png');
const imageBuffer = fs.readFileSync(testImagePath);
const base64Image = `data:image/png;base64,${imageBuffer.toString('base64')}`;

async function testSimple() {
  console.log('Testing simple API call...');
  
  try {
    const response = await fetch('http://localhost:3000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: [{
          id: '1',
          role: 'user',
          content: 'Hello'
        }],
        processingMode: 'Auto',
        aiModel: 'GPT',
        imageData: base64Image,
        useOriginalImage: true
      }),
    });
    
    console.log('Response status:', response.status);
    console.log('Response ok:', response.ok);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.log('Error response:', errorText);
    } else {
      console.log('✅ Chat API works');
    }
  } catch (error) {
    console.log('❌ Error:', error.message);
  }
}

testSimple().catch(console.error); 