from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
import os
import logging
import json
import traceback
from detection_model import load_detection_model, run_object_detection_with_boxes
from segmentation_inference import load_segmentation_model, run_segmentation_with_masks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
class_names = ['Anthracnose', 'Bacterial Wilt', 'Downy-mildew', 'Fresh', 'Gummy Stem Blight', 'Pawdery-mildew']
img_size = (224, 224)


def load_model():
    global model
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, 'config.json')
        weights_path = os.path.join(base_dir, 'model.weights.h5')

        with open(config_path, 'r') as f:
            config = json.load(f)
        model = tf.keras.models.model_from_json(json.dumps(config))
        logger.info("✅ Model architecture loaded from config.json")

        model.load_weights(weights_path)
        logger.info("✅ Model weights loaded from model.weights.h5")

        return True
    except Exception as e:
        logger.error(f"❌ Failed to load classification model: {str(e)}")
        return False


def preprocess_image(image_data):
    try:
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            image = Image.open(io.BytesIO(image_data))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize(img_size)
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = tf.keras.applications.mobilenet_v3.preprocess_input(image_array * 255.0)
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {str(e)}")
        return None


def make_prediction(image_array):
    try:
        predictions = model.predict(image_array)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        class_probabilities = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}

        return {
            'predicted_class': predicted_class_name,
            'predicted_class_index': int(predicted_class_idx),
            'confidence': confidence,
            'all_probabilities': class_probabilities
        }

    except Exception as e:
        logger.error("❌ Error making prediction")
        traceback.print_exc()
        return None


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded', 'success': False}), 500

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided', 'success': False}), 400

        image_array = preprocess_image(data['image'])
        if image_array is None:
            return jsonify({'error': 'Failed to preprocess image', 'success': False}), 400

        prediction_result = make_prediction(image_array)
        if prediction_result is None:
            return jsonify({'error': 'Failed to make prediction', 'success': False}), 500

        return jsonify({'success': True, 'prediction': prediction_result})

    except Exception as e:
        logger.error("❌ Error in /predict endpoint")
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}', 'success': False}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded', 'success': False}), 500

        data = request.get_json()
        if not data or 'images' not in data or not isinstance(data['images'], list):
            return jsonify({'error': 'Invalid images input', 'success': False}), 400

        predictions = []
        for i, image_data in enumerate(data['images']):
            image_array = preprocess_image(image_data)
            if image_array is None:
                predictions.append({'index': i, 'error': 'Preprocessing failed', 'success': False})
                continue
            prediction_result = make_prediction(image_array)
            if prediction_result is None:
                predictions.append({'index': i, 'error': 'Prediction failed', 'success': False})
                continue
            predictions.append({'index': i, 'success': True, 'prediction': prediction_result})

        return jsonify({'success': True, 'predictions': predictions})

    except Exception as e:
        logger.error(f"❌ Error in predict_batch endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}', 'success': False}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded', 'success': False}), 500

        return jsonify({
            'success': True,
            'model_info': {
                'input_shape': model.input_shape,
                'output_shape': model.output_shape,
                'num_classes': len(class_names),
                'class_names': class_names,
                'image_size': img_size
            }
        })

    except Exception as e:
        logger.error(f"❌ Error in model_info endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}', 'success': False}), 500


@app.route('/predict_detection', methods=['POST'])
def predict_detection():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided', 'success': False}), 400

        result = run_object_detection_with_boxes(data['image'])
        if not result['success']:
            return jsonify({'error': result['error'], 'success': False}), 500

        return jsonify({
            'success': True,
            'detections': result['detections'],
            'annotated_image': result['annotated_image']
        })

    except Exception as e:
        logger.error(f"❌ Exception in /predict_detection: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500


# @app.route('/predict_segmentation', methods=['POST'])
# def predict_segmentation():
#     try:
#         data = request.get_json()
#         if not data or 'image' not in data:
#             return jsonify({'error': 'No image data provided', 'success': False}), 400

#         result = run_segmentation_with_masks(data['image'])
#         if not result['success']:
#             return jsonify({'error': result.get('error', 'Segmentation failed'), 'success': False}), 500

#         return jsonify({
#             'success': True,
#             'detections': result['detections'],
#             'annotated_image': result['annotated_image'],
#             'mask': result['mask']
#         })

#     except Exception as e:
#         logger.error(f"❌ Exception in /predict_segmentation: {str(e)}")
#         return jsonify({'error': str(e), 'success': False}), 500

@app.route('/predict_segmentation', methods=['POST'])
def predict_segmentation():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided', 'success': False}), 400

        logger.info("📸 Starting segmentation inference...")
        result = run_segmentation_with_masks(data['image'])
        
        if not result['success']:
            logger.error(f"❌ Segmentation failed: {result.get('error', 'Unknown error')}")
            return jsonify({'error': result.get('error', 'Segmentation failed'), 'success': False}), 500

        # Log the sizes of returned data
        response_data = {
            'success': True,
            'detections': result['detections'],
            'annotated_image': result['annotated_image'],
            'mask': result['mask']
        }
        
        # Log data sizes for debugging
        logger.info(f"📊 Response data sizes:")
        logger.info(f"   - Detections: {len(str(result['detections']))} chars")
        logger.info(f"   - Annotated image: {len(result['annotated_image']) if result['annotated_image'] else 0} chars")
        logger.info(f"   - Mask: {len(result['mask']) if result['mask'] else 0} chars")
        
        # Check if images are properly base64 encoded
        if result['annotated_image']:
            logger.info(f"   - Annotated image starts with: {result['annotated_image'][:50]}...")
        if result['mask']:
            logger.info(f"   - Mask starts with: {result['mask'][:50]}...")
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"❌ Exception in /predict_segmentation: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


# === App entry point ===
if __name__ == '__main__':
    model_loaded = load_model()
    logger.info(f"Model loaded: {model_loaded}")

    detection_model_loaded = load_detection_model()
    logger.info(f"Detection model loaded: {detection_model_loaded}")

    segmentation_model_loaded = load_segmentation_model()
    logger.info(f"Segmentation model loaded: {segmentation_model_loaded}")

    if model_loaded and detection_model_loaded and segmentation_model_loaded:
        logger.info("🚀 Starting Flask API server on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("❌ Failed to load one or more models. Exiting...")
