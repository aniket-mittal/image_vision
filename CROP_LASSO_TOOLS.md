# Crop and Lasso Tools

## Overview

The crop and lasso tools allow users to select specific regions of an image for focused analysis. When a user selects one of these tools and draws a selection, the system:

1. Reverts to the original unmasked image
2. Generates a caption of the image
3. Creates a masked version based on the user's selection
4. Provides the masked image to GPT-4V for analysis

## How It Works

### Crop Tool
- **Selection**: Rectangular region selection
- **Masking**: Keeps the selected rectangle clear, blurs the rest of the image
- **Use Case**: Focus on specific objects or areas within the image

### Lasso Tool
- **Selection**: Free-form path selection (closed loop)
- **Masking**: Keeps the area inside the lasso path clear, blurs everything outside
- **Use Case**: Select irregularly shaped objects or regions

## Implementation Details

### Frontend Changes (`frontend/app/page.tsx`)

1. **Tool Selection**: When crop or lasso tools are clicked, the system reverts to the original image
2. **Drawing Interface**: Canvas overlay for drawing selections
3. **Visual Feedback**: Instructions overlay and selection indicators
4. **API Integration**: Automatically calls the crop-lasso-process API when a selection is made

### Backend API (`frontend/app/api/crop-lasso-process/route.ts`)

1. **Image Captioning**: Uses GPT-4V to generate a description of the image
2. **Mask Creation**: Server-side image processing using the `canvas` library
3. **Analysis**: Sends the masked image to GPT-4V for focused analysis

## API Endpoint

### POST `/api/crop-lasso-process`

**Request Body:**
```json
{
  "imageData": "data:image/jpeg;base64,...",
  "userQuery": "What can you see in the selected region?",
  "toolType": "crop" | "lasso",
  "coordinates": [x1, y1, x2, y2, ...]
}
```

**Response:**
```json
{
  "processedImageData": "data:image/jpeg;base64,...",
  "refinedQuery": "User selected crop region",
  "processingType": "crop",
  "parameters": {
    "coordinates": [100, 100, 200, 200],
    "toolType": "crop"
  },
  "answer": "Analysis of the selected region..."
}
```

## User Experience

1. **Upload Image**: User uploads an image to analyze
2. **Select Tool**: User clicks on crop or lasso tool
3. **Draw Selection**: User draws their selection on the image
4. **Automatic Processing**: System creates masked image and analyzes it
5. **Get Answer**: User receives focused analysis of the selected region

## Technical Features

- **Real-time Drawing**: Live preview of selection while drawing
- **Visual Instructions**: Clear guidance on how to use each tool
- **Selection Management**: Ability to clear selections and start over
- **Error Handling**: Graceful fallback to original image if processing fails
- **TypeScript Support**: Full type safety throughout the implementation

## Dependencies

- `canvas`: For server-side image processing
- `next`: For API routes and frontend framework
- `lucide-react`: For tool icons

## Testing

Run the test file to verify API functionality:
```bash
node test_crop_lasso.js
```

## Future Enhancements

- Multiple selection support
- Selection history
- Undo/redo functionality
- Custom blur intensity
- Selection templates
- Batch processing of multiple selections 