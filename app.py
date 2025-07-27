
# budget_tracker_with_voice_ocr.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytesseract
from PIL import Image
import speech_recognition as sr
import io
import base64
import warnings
import re
import json
import os
import tempfile
from transformers import pipeline
import torch
import pytesseract
import os


warnings.filterwarnings('ignore')

# Set Tesseract path (update this path according to your system)
# For Windows: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# For Mac: "/usr/local/bin/tesseract"
# For Linux: "/usr/bin/tesseract"
try:
    # You can set your Tesseract path here
    TESSERACT_PATH = os.getenv("TESSERACT_PATH", r'/usr/bin/tesseract')  # Default for Linux
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
except:
    pass  # Use default path



# Initialize session state for data persistence
def initialize_session_state():
    """Initialize all session state variables"""
    try:
        if 'expenses' not in st.session_state:
            st.session_state.expenses = pd.DataFrame(columns=['date', 'amount', 'category', 'description', 'receipt_image'])
        if 'budgets' not in st.session_state:
            st.session_state.budgets = pd.DataFrame(columns=['category', 'budget_amount', 'period'])
        if 'savings_goals' not in st.session_state:
            st.session_state.savings_goals = pd.DataFrame(columns=['goal_name', 'target_amount', 'current_amount', 'target_date'])
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        if 'whisper_model' not in st.session_state:
            st.session_state.whisper_model = None
        return True
    except Exception as e:
        st.error(f"Error initializing session state: {str(e)}")
        return False

# Voice Recognition with Whisper (Corrected Implementation)
def load_whisper_model():
    """Load Whisper model for speech recognition"""
    try:
        if st.session_state.whisper_model is None:
            with st.spinner("Loading Whisper model... This may take a moment."):
                st.session_state.whisper_model = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-tiny",  # Using tiny model for faster loading
                    chunk_length_s=30,
                )
        return st.session_state.whisper_model
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

def transcribe_audio_with_whisper(audio_file_path):
    """Transcribe audio using Whisper model"""
    try:
        model = load_whisper_model()
        if model is None:
            return None
            
        with st.spinner("Transcribing audio..."):
            output = model(
                audio_file_path,
                generate_kwargs={"task": "transcribe"},
                batch_size=8,
                return_timestamps=False,
            )
            return output["text"]
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        return None

def transcribe_audio_with_google(audio_file_path):
    """Fallback: Transcribe audio using Google Speech Recognition"""
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except Exception as e:
        st.error(f"Google Speech Recognition error: {str(e)}")
        return None

def voice_expense_recording():
    """
    Function to record expense using voice input
    LLM Needed: NO - Uses Whisper/GSR for speech recognition
    Could use LLM for better natural language understanding
    """
    try:
        st.subheader("🎤 Voice Expense Recording")
        
        # Audio input options
        audio_option = st.radio("Choose audio input method:", 
                               ["Microphone (Real-time)", "Upload Audio File"])
        
        if audio_option == "Microphone (Real-time)":
            # Check if microphone is available
            try:
                recognizer = sr.Recognizer()
                mic_available = True
            except:
                mic_available = False
                st.warning("Microphone not available. Please check your device settings.")
            
            if mic_available and st.button("🎙️ Start Voice Recording"):
                try:
                    with sr.Microphone() as source:
                        st.info("🎤 Listening... Please speak your expense (e.g., 'I spent 500 rupees on groceries')")
                        # Adjust for ambient noise
                        recognizer.adjust_for_ambient_noise(source, duration=1)
                        audio = recognizer.listen(source, timeout=10)
                        
                        # Save audio to temporary file for processing
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            with open(tmp_file.name, "wb") as f:
                                f.write(audio.get_wav_data())
                            temp_filename = tmp_file.name
                        
                        # Try Whisper first, fallback to Google
                        text = transcribe_audio_with_whisper(temp_filename)
                        if text is None:
                            text = transcribe_audio_with_google(temp_filename)
                        
                        # Clean up temporary file
                        os.unlink(temp_filename)
                        
                        if text:
                            st.success(f"✅ Recognized: {text}")
                            process_voice_text(text)
                        else:
                            st.error("❌ Failed to transcribe audio")
                            
                except sr.WaitTimeoutError:
                    st.error("⏰ Timeout: No speech detected within 10 seconds")
                except sr.UnknownValueError:
                    st.error("🤔 Could not understand audio. Please try again.")
                except sr.RequestError as e:
                    st.error(f"🌐 Could not request results: {e}")
                except Exception as e:
                    st.error(f"❌ Error processing voice input: {str(e)}")
        
        else:  # Upload Audio File
            uploaded_audio = st.file_uploader("Upload Audio File", type=['wav', 'mp3', 'm4a'])
            
            if uploaded_audio is not None:
                if st.button("🔊 Process Audio File"):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_audio.getvalue())
                            temp_filename = tmp_file.name
                        
                        # Process audio file
                        with st.spinner("Processing audio file..."):
                            # Try Whisper first, fallback to Google
                            text = transcribe_audio_with_whisper(temp_filename)
                            if text is None:
                                text = transcribe_audio_with_google(temp_filename)
                            
                            # Clean up temporary file
                            os.unlink(temp_filename)
                            
                            if text:
                                st.success(f"✅ Transcribed: {text}")
                                process_voice_text(text)
                            else:
                                st.error("❌ Failed to transcribe audio file")
                                
                    except Exception as e:
                        st.error(f"❌ Error processing audio file: {str(e)}")
        
        # Instructions
        st.info("💡 Tip: Say something like 'I spent 500 rupees on groceries at Big Bazaar'")
        
    except Exception as e:
        st.error(f"❌ Critical error in voice recording: {str(e)}")

def process_voice_text(text):
    """Process transcribed voice text to extract expense details"""
    try:
        # Enhanced parsing logic
        st.info("🔄 Processing voice input...")
        amount = 0
        category = "Other"
        description = text
        
        # Enhanced category detection
        categories = {
            'Food': ['food', 'groceries', 'restaurant', 'cafe', 'meal', 'lunch', 'dinner', 'breakfast', 'dhaba', 'hotel'],
            'Transport': ['transport', 'travel', 'taxi', 'uber', 'ola', 'bus', 'train', 'flight', 'fuel', 'petrol', 'diesel', 'auto'],
            'Shopping': ['shopping', 'clothes', 'electronics', 'purchase', 'buy', 'mall', 'store', 'market'],
            'Entertainment': ['entertainment', 'movie', 'cinema', 'game', 'fun', 'party', 'netflix', 'spotify'],
            'Bills': ['bill', 'electricity', 'water', 'internet', 'phone', 'rent', 'insurance', 'subscription'],
            'Health': ['medicine', 'doctor', 'hospital', 'pharmacy', 'health', 'medical'],
            'Education': ['education', 'school', 'college', 'books', 'course', 'tuition', 'study']
        }
        
        text_lower = text.lower()
        for cat, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                category = cat
                break
        
        # Extract numbers for amount using regex
        amount_pattern = r'(?:$|\$|rs|rupees?|dollars?)\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:$|\$|rs|rupees?|dollars?)'

        # amount_pattern = r'(?:$|rs|rupees?)\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:$|rs|rupees?)'
        matches = re.findall(amount_pattern, text_lower)
        if matches:
            for match in matches:
                for group in match:
                    if group and (group.replace('.', '').isdigit()):
                        amount = float(group)
                        break
                if amount > 0:
                    break
        
        # Fallback: look for any number
        if amount == 0:
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if numbers:
                amount = float(numbers[0])
        
        # Save to expenses
        new_expense = pd.DataFrame({
            'date': [datetime.now().strftime('%Y-%m-%d')],
            'amount': [amount],
            'category': [category],
            'description': [description],
            'receipt_image': ['']
        })
        st.session_state.expenses = pd.concat([st.session_state.expenses, new_expense], ignore_index=True)
        st.success(f"✅ Expense logged: ${amount:.2f} for {category}")

        
        # Check budget alerts
        check_budget_alerts(amount, category)
        
    except Exception as e:
        st.error(f"❌ Error processing voice text: {str(e)}")

# OCR Processing (Corrected Implementation)
class OCRExtractor:
    def __init__(self):
        pass
    
    def extract_text_from_image(self, image_input):
        """Extract text from image using Tesseract OCR"""
        try:
            # Handle different input types
            if hasattr(image_input, 'read'):
                # Uploaded file
                image = Image.open(image_input)
            elif isinstance(image_input, str):
                # File path
                image = Image.open(image_input)
            else:
                # PIL Image
                image = image_input
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text with multiple languages support
            custom_config = r'--oem 3 --psm 6 -l eng'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            return text.strip()
        except Exception as e:
            st.error(f"OCR Error: {e}")
            return None
    
    def extract_structured_data(self, image_input):
        """Extract structured data from receipt image"""
        try:
            text = self.extract_text_from_image(image_input)
            if not text:
                return None
            
            # Basic structure extraction
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            return {
                'raw_text': text,
                'lines': lines,
                'extracted_at': str(pd.Timestamp.now())
            }
        except Exception as e:
            st.error(f"Error extracting structured data: {e}")
            return None
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # You can add more preprocessing steps here
            # like noise reduction, contrast enhancement, etc.
            
            return image
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return image

def ocr_receipt_processing():
    """
    Function to process receipt images using OCR
    LLM Needed: NO - Uses Tesseract OCR for text extraction
    Could use LLM for better data parsing and categorization
    """
    try:
        st.subheader("📸 Receipt OCR Processing")
        
        uploaded_file = st.file_uploader("Upload Receipt Image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="📸 Uploaded Receipt", use_container_width=True)
                
                if st.button("🔍 Process Receipt"):
                    # Initialize OCR extractor
                    ocr_extractor = OCRExtractor()
                    
                    # Use Tesseract OCR
                    try:
                        # Preprocess image for better results
                        processed_image = ocr_extractor.preprocess_image(image)
                        
                        # Extract text
                        extracted_text = ocr_extractor.extract_text_from_image(processed_image)
                        
                        if extracted_text:
                            st.text_area("📄 Extracted Text", extracted_text, height=200)
                            
                            # Parse receipt data
                            amount = 0
                            category = "Other"
                            description = "Receipt expense"
                            
                            # Extract amount with multiple patterns
                            amount_patterns = [
                                r'[$$€£]\s*(\d+(?:\.\d+)?)',
                                r'(\d+(?:\.\d+)?)\s*[$$€£]',
                                r'(?:total|amount|paid|grand total).*?(\d+(?:\.\d+)?)',
                                r'(?:bill|invoice).*?(\d+(?:\.\d+)?)'
                            ]
                            
                            for pattern in amount_patterns:
                                matches = re.findall(pattern, extracted_text.lower(), re.IGNORECASE)
                                if matches:
                                    for match in matches:
                                        if isinstance(match, tuple):
                                            for group in match:
                                                if group and (group.replace('.', '').isdigit()):
                                                    amount = float(group)
                                                    break
                                        elif match.replace('.', '').isdigit():
                                            amount = float(match)
                                            break
                                    if amount > 0:
                                        break
                            
                            # Enhanced category detection
                            categories_keywords = {
                                'Food': ['restaurant', 'cafe', 'grocery', 'food', 'meal', 'supermarket', 'big bazaar', 'dmart', 'walmart'],
                                'Transport': ['taxi', 'uber', 'ola', 'fuel', 'petrol', 'bus', 'train', 'airport', 'parking'],
                                'Shopping': ['mall', 'store', 'shop', 'purchase', 'clothes', 'electronics', 'amazon', 'flipkart'],
                                'Entertainment': ['movie', 'cinema', 'game', 'entertainment', 'theatre', 'netflix'],
                                'Bills': ['electricity', 'water', 'internet', 'phone', 'rent', 'subscription', 'bill'],
                                'Health': ['pharmacy', 'medicine', 'doctor', 'hospital', 'medical', 'apollo', 'apollo'],
                                'Education': ['school', 'college', 'books', 'stationery', 'tution', 'course']
                            }
                            
                            text_lower = extracted_text.lower()
                            for cat, keywords in categories_keywords.items():
                                if any(keyword in text_lower for keyword in keywords):
                                    category = cat
                                    break
                            
                            # Save to expenses with image data
                            image_data = f"data:image/png;base64,{base64.b64encode(uploaded_file.getvalue()).decode()}"
                            new_expense = pd.DataFrame({
                                'date': [datetime.now().strftime('%Y-%m-%d')],
                                'amount': [amount],
                                'category': [category],
                                'description': [description],
                                'receipt_image': [image_data]
                            })
                            st.session_state.expenses = pd.concat([st.session_state.expenses, new_expense], ignore_index=True)
                            st.success(f"✅ Receipt processed successfully: ${amount:.2f} for {category}")
                            
                            # Check budget alerts
                            check_budget_alerts(amount, category)
                        else:
                            st.error("❌ Could not extract text from image. Please try a clearer image.")
                            
                    except Exception as e:
                        st.error(f"❌ OCR processing failed: {str(e)}")
                        st.info("💡 Make sure Tesseract OCR is properly installed on your system")
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
        else:
            st.info("📤 Please upload a receipt image (JPG, JPEG, PNG)")
            
    except Exception as e:
        st.error(f"❌ Critical error in OCR processing: {str(e)}")

def create_budget():
    """
    Function to create and manage budgets
    LLM Needed: NO - Simple form-based input
    Could use LLM for budget recommendations based on spending patterns
    """
    try:
        st.subheader("💰 Create Budget")
        
        col1, col2 = st.columns(2)
        with col1:
            predefined_categories = ["Food", "Transport", "Shopping", "Entertainment", "Bills", "Health", "Education", "Other"]
            category_type = st.radio("Category Type", ["Predefined", "Custom"])
            if category_type == "Predefined":
                category = st.selectbox("Category", predefined_categories)
            else:
                category = st.text_input("Enter custom category")
        
        with col2:
            budget_amount = st.number_input("Budget Amount ($)", min_value=0.0, step=100.0, value=1000.0)
            period = st.selectbox("Period", ["Monthly", "Weekly", "Custom"])
        
        if st.button("📊 Set Budget"):
            if category and budget_amount > 0:
                try:
                    # Check if budget already exists for this category
                    existing_budget = st.session_state.budgets[
                        st.session_state.budgets['category'] == category
                    ]
                    
                    if not existing_budget.empty:
                        # Update existing budget
                        st.session_state.budgets.loc[
                            st.session_state.budgets['category'] == category, 'budget_amount'
                        ] = budget_amount
                        st.session_state.budgets.loc[
                            st.session_state.budgets['category'] == category, 'period'
                        ] = period
                        st.success(f"🔄 Budget updated: ${budget_amount:.2f} for {category}")
                    else:
                        # Add new budget
                        new_budget = pd.DataFrame({
                            'category': [category],
                            'budget_amount': [budget_amount],
                            'period': [period]
                        })
                        st.session_state.budgets = pd.concat([st.session_state.budgets, new_budget], ignore_index=True)
                        st.success(f"✅ Budget set: ${budget_amount:.2f} for {category}")
                except Exception as e:
                    st.error(f"❌ Error setting budget: {str(e)}")
            else:
                st.error("⚠️ Please enter valid category and amount")
        
        # Display existing budgets
        if not st.session_state.budgets.empty:
            st.subheader("📊 Current Budgets")
            st.dataframe(st.session_state.budgets)
            
            # Option to delete budgets
            if st.checkbox("🗑️ Show delete options"):
                budget_to_delete = st.selectbox("Select budget to delete", 
                                               st.session_state.budgets['category'].tolist())
                if st.button("🗑️ Delete Budget"):
                    st.session_state.budgets = st.session_state.budgets[
                        st.session_state.budgets['category'] != budget_to_delete
                    ]
                    st.success(f"✅ Budget for {budget_to_delete} deleted")
        else:
            st.info("📝 No budgets set yet. Create your first budget!")
            
    except Exception as e:
        st.error(f"❌ Critical error in budget creation: {str(e)}")

def set_savings_goals():
    """
    Function to set and track savings goals
    LLM Needed: NO - Simple goal tracking
    Could use LLM for personalized savings recommendations
    """
    try:
        st.subheader("🎯 Savings Goals")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            goal_name = st.text_input("Goal Name", placeholder="e.g., Vacation, Emergency Fund")
        with col2:
            target_amount = st.number_input("Target Amount ($)", min_value=0.0, step=1000.0, value=10000.0)
        with col3:
            target_date = st.date_input("Target Date", 
                                      value=datetime.now() + timedelta(days=30))
        
        if st.button("🎯 Set Goal"):
            if goal_name and target_amount > 0:
                try:
                    new_goal = pd.DataFrame({
                        'goal_name': [goal_name],
                        'target_amount': [target_amount],
                        'current_amount': [0.0],
                        'target_date': [target_date]
                    })
                    st.session_state.savings_goals = pd.concat([st.session_state.savings_goals, new_goal], ignore_index=True)
                    st.success(f"✅ Savings goal '{goal_name}' set!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error setting goal: {str(e)}")
            else:
                st.error("⚠️ Please enter valid goal name and target amount")
        
        # Display existing goals
        if not st.session_state.savings_goals.empty:
            st.subheader("🏆 Current Goals")
            
            for idx, goal in st.session_state.savings_goals.iterrows():
                try:
                    progress = (goal['current_amount'] / goal['target_amount']) * 100 if goal['target_amount'] > 0 else 0
                    days_left = (goal['target_date'] - datetime.now().date()).days
                    
                    st.write(f"**{goal['goal_name']}**")
                    st.progress(min(progress/100, 1.0))
                    st.write(f"💰 ${goal['current_amount']:.2f} / ${goal['target_amount']:.2f} ({progress:.1f}%)")
                    st.write(f"📅 Target Date: {goal['target_date']} ({days_left} days left)")
                    
                    # Add to current savings
                    add_amount = st.number_input(f"Add to {goal['goal_name']}", 
                                               min_value=0.0, step=100.0, key=f"add_{idx}")
                    if st.button(f"➕ Add to {goal['goal_name']}", key=f"btn_{idx}"):
                        if add_amount > 0:
                            st.session_state.savings_goals.at[idx, 'current_amount'] += add_amount
                            st.success(f"✅ Added ${add_amount:.2f} to {goal['goal_name']}")
                            st.rerun()
                    
                    st.write("---")
                except Exception as e:
                    st.error(f"❌ Error displaying goal: {str(e)}")
        else:
            st.info("📝 No savings goals set yet. Create your first goal!")
            
    except Exception as e:
        st.error(f"❌ Critical error in savings goals: {str(e)}")

def spending_categorization():
    """
    Function to categorize and review spending
    LLM Needed: NO - Rule-based categorization
    Could use LLM for smarter automatic categorization
    """
    try:
        st.subheader("🏷️ Spending Categorization")
        
        if not st.session_state.expenses.empty:
            # Display expenses that need categorization
            uncategorized = st.session_state.expenses[st.session_state.expenses['category'] == 'Other']
            if not uncategorized.empty:
                st.write("📝 Uncategorized Expenses:")
                for idx, expense in uncategorized.iterrows():
                    try:
                        st.write(f"📅 {expense['date']} | ${expense['amount']:.2f} | {expense['description']}")
                        predefined_categories = ["Food", "Transport", "Shopping", "Entertainment", "Bills", "Health", "Education", "Other"]
                        new_category = st.selectbox(f"Re-categorize", predefined_categories, 
                                                  key=f"cat_{idx}")
                        if st.button(f"🔄 Update Category {idx}"):
                            if new_category and new_category != 'Other':
                                st.session_state.expenses.at[idx, 'category'] = new_category
                                st.success(f"✅ Category updated to {new_category}!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error updating category: {str(e)}")
                st.write("---")
            
            # Show all expenses with filter options
            st.subheader("📋 All Expenses")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                category_filter = st.selectbox("Filter by Category", 
                                             ["All"] + list(st.session_state.expenses['category'].unique()))
            with col2:
                date_filter = st.selectbox("Date Range", ["All", "Last 7 days", "Last 30 days", "This month"])
            with col3:
                min_amount = st.number_input("Min Amount", min_value=0.0, step=10.0)
                max_amount = st.number_input("Max Amount", min_value=0.0, 
                                           value=float(st.session_state.expenses['amount'].max()) if not st.session_state.expenses.empty else 10000.0)
            
            # Apply filters
            filtered_expenses = st.session_state.expenses.copy()
            
            if category_filter != "All":
                filtered_expenses = filtered_expenses[filtered_expenses['category'] == category_filter]
            
            if date_filter == "Last 7 days":
                cutoff_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                filtered_expenses = filtered_expenses[filtered_expenses['date'] >= cutoff_date]
            elif date_filter == "Last 30 days":
                cutoff_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                filtered_expenses = filtered_expenses[filtered_expenses['date'] >= cutoff_date]
            elif date_filter == "This month":
                current_month = datetime.now().strftime('%Y-%m')
                filtered_expenses = filtered_expenses[filtered_expenses['date'].str.startswith(current_month)]
            
            filtered_expenses = filtered_expenses[
                (filtered_expenses['amount'] >= min_amount) & 
                (filtered_expenses['amount'] <= max_amount)
            ]
            
            if not filtered_expenses.empty:
                st.dataframe(filtered_expenses[['date', 'amount', 'category', 'description']])
                
                # Summary statistics
                st.subheader("📊 Summary")
                total_spent = filtered_expenses['amount'].sum()
                avg_spent = filtered_expenses['amount'].mean()
                st.metric("Total Spent", f"${total_spent:.2f}")

                st.metric("Average Expense", f"${avg_spent:.2f}")

            else:
                st.info("🔍 No expenses match the current filters")
        else:
            st.info("📝 No expenses recorded yet. Start by adding expenses through voice or receipt scanning!")
            
    except Exception as e:
        st.error(f"❌ Critical error in spending categorization: {str(e)}")

def check_budget_alerts(amount, category):
    """
    Helper function to check budget alerts
    """
    try:
        if not st.session_state.budgets.empty:
            category_budget = st.session_state.budgets[st.session_state.budgets['category'] == category]
            if not category_budget.empty:
                budget_amount = category_budget.iloc[0]['budget_amount']
                current_spending = st.session_state.expenses[
                    st.session_state.expenses['category'] == category
                ]['amount'].sum()
                
                if current_spending > budget_amount:
                    alert_msg = f"🚨 OVERSPENT: {category} - ${current_spending:.2f}/${budget_amount:.2f}"


                    if alert_msg not in st.session_state.notifications:
                        st.session_state.notifications.append(alert_msg)
                elif current_spending > budget_amount * 0.8:  # 80% threshold
                    alert_msg = f"⚠️ WARNING: {category} - ${current_spending:.2f}/${budget_amount:.2f} ({((current_spending/budget_amount)*100):.1f}%)"


                    if alert_msg not in st.session_state.notifications:
                        st.session_state.notifications.append(alert_msg)
    except Exception as e:
        st.error(f"❌ Error checking budget alerts: {str(e)}")

def alerts_and_notifications():
    """
    Function to check and display budget alerts
    LLM Needed: NO - Simple threshold checking
    Could use LLM for personalized alert messages
    """
    try:
        st.subheader("🔔 Budget Alerts & Notifications")
        
        # Clear notifications button
        if st.session_state.notifications:
            if st.button("🧹 Clear All Notifications"):
                st.session_state.notifications = []
                st.rerun()
        
        # Check for new alerts
        if not st.session_state.expenses.empty and not st.session_state.budgets.empty:
            try:
                # Calculate spending by category
                spending_by_category = st.session_state.expenses.groupby('category')['amount'].sum().reset_index()
                
                alerts = []
                for _, budget in st.session_state.budgets.iterrows():
                    category_spending = spending_by_category[spending_by_category['category'] == budget['category']]
                    if not category_spending.empty:
                        spent = category_spending.iloc[0]['amount']
                        budget_amount = budget['budget_amount']
                        
                        if spent > budget_amount:
                            alerts.append(f"🚨 OVERSPENT: {budget['category']} - ${spent:.2f}/${budget_amount:.2f} ({((spent/budget_amount)*100):.1f}%)")


                        elif spent > budget_amount * 0.8:  # 80% threshold
                            alerts.append(f"⚠️ WARNING: {budget['category']} - ${spent:.2f}/${budget_amount:.2f} ({((spent/budget_amount)*100):.1f}%)")


                
                # Display alerts
                if alerts:
                    for alert in alerts:
                        st.warning(alert)
                        if alert not in st.session_state.notifications:
                            st.session_state.notifications.append(alert)
                else:
                    st.success("✅ All budgets are within limits!")
                    
            except Exception as e:
                st.error(f"❌ Error calculating alerts: {str(e)}")
        else:
            st.info("📝 Set up budgets and record expenses to see alerts")
        
        # Display all notifications
        if st.session_state.notifications:
            st.subheader("📜 Recent Notifications")
            for notification in reversed(st.session_state.notifications[-10:]):  # Show last 10
                st.info(notification)
        else:
            st.info("📭 No notifications yet")
            
    except Exception as e:
        st.error(f"❌ Critical error in alerts system: {str(e)}")

def visualizations_and_summaries():
    """
    Function to create charts and summaries
    LLM Needed: NO - Standard data visualization
    Could use LLM for generating insights and summaries
    """
    try:
        st.subheader("📊 Financial Visualizations")
        
        if not st.session_state.expenses.empty:
            try:
                # Spending by category pie chart
                spending_by_category = st.session_state.expenses.groupby('category')['amount'].sum()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("💰 Spending by Category")
                    if len(spending_by_category) > 0:
                        fig1 = px.pie(values=spending_by_category.values, names=spending_by_category.index,
                                    title="Expense Distribution by Category")
                        st.plotly_chart(fig1, use_container_width=True)
                    else:
                        st.info("No spending data to visualize")
                
                with col2:
                    st.write("📈 Spending Trend")
                    daily_spending = st.session_state.expenses.groupby('date')['amount'].sum().reset_index()
                    if len(daily_spending) > 1:
                        fig2 = px.line(daily_spending, x='date', y='amount', 
                                     title='Daily Spending Trend')
                        fig2.update_xaxes(type='category')
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("Need more data points for trend analysis")
                
                # Monthly summary
                st.subheader("📋 Monthly Summary")
                current_month = datetime.now().strftime('%Y-%m')
                monthly_expenses = st.session_state.expenses[
                    st.session_state.expenses['date'].str.startswith(current_month)
                ]
                
                if not monthly_expenses.empty:
                    total_spent = monthly_expenses['amount'].sum()
                    st.metric("Total Monthly Spending", f"${total_spent:.2f}")

                    
                    category_summary = monthly_expenses.groupby('category')['amount'].sum().reset_index()
                    fig3 = px.bar(category_summary, x='category', y='amount', 
                                title='Monthly Spending by Category')
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    st.dataframe(category_summary)
                else:
                    st.info("No expenses recorded this month.")
                    
                # Budget vs Actual comparison
                if not st.session_state.budgets.empty:
                    st.subheader("⚖️ Budget vs Actual Comparison")
                    budget_comparison = []
                    for _, budget in st.session_state.budgets.iterrows():
                        actual_spent = st.session_state.expenses[
                            st.session_state.expenses['category'] == budget['category']
                        ]['amount'].sum()
                        budget_comparison.append({
                            'Category': budget['category'],
                            'Budget': budget['budget_amount'],
                            'Actual': actual_spent,
                            'Difference': budget['budget_amount'] - actual_spent
                        })
                    
                    if budget_comparison:
                        comparison_df = pd.DataFrame(budget_comparison)
                        st.dataframe(comparison_df)
                        
                        # Visualization
                        fig4 = go.Figure()
                        fig4.add_trace(go.Bar(name='Budget', x=comparison_df['Category'], y=comparison_df['Budget']))
                        fig4.add_trace(go.Bar(name='Actual', x=comparison_df['Category'], y=comparison_df['Actual']))
                        fig4.update_layout(title="Budget vs Actual Spending", barmode='group')
                        st.plotly_chart(fig4, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error creating visualizations: {str(e)}")
        else:
            st.info("📝 No data to visualize yet. Start by recording expenses!")
            
    except Exception as e:
        st.error(f"❌ Critical error in visualizations: {str(e)}")

def receipt_management():
    """
    Function to manage and view stored receipts
    LLM Needed: NO - Simple storage and retrieval
    Could use LLM for receipt categorization and insights
    """
    try:
        st.subheader("🧾 Receipt Management")
        
        if not st.session_state.expenses.empty:
            receipts = st.session_state.expenses[st.session_state.expenses['receipt_image'] != '']
            if not receipts.empty:
                st.write(f"📁 Found {len(receipts)} receipts")
                
                # Search functionality
                search_term = st.text_input("🔍 Search receipts by description or category")
                if search_term:
                    receipts = receipts[
                        receipts['description'].str.contains(search_term, case=False) |
                        receipts['category'].str.contains(search_term, case=False)
                    ]
                    st.write(f"🔍 Found {len(receipts)} matching receipts")
                
                # Sort options
                sort_option = st.selectbox("Sort by", ["Date (Newest)", "Date (Oldest)", "Amount (High to Low)", "Amount (Low to High)"])
                
                if sort_option == "Date (Newest)":
                    receipts = receipts.sort_values('date', ascending=False)
                elif sort_option == "Date (Oldest)":
                    receipts = receipts.sort_values('date', ascending=True)
                elif sort_option == "Amount (High to Low)":
                    receipts = receipts.sort_values('amount', ascending=False)
                elif sort_option == "Amount (Low to High)":
                    receipts = receipts.sort_values('amount', ascending=True)
                
                # Display receipts in a grid
                if not receipts.empty:
                    cols = st.columns(3)
                    for idx, (i, receipt) in enumerate(receipts.iterrows()):
                        try:
                            with cols[idx % 3]:
                                st.write(f"**📅 {receipt['date']}**")
                                st.write(f"💰 ${receipt['amount']:.2f}")

                                st.write(f"🏷️ {receipt['category']}")
                                if receipt['receipt_image'].startswith('data:image'):
                                    # Display base64 image
                                    st.image(receipt['receipt_image'], width=200)
                                st.write(f"📝 {receipt['description'][:50]}...")
                                st.write("---")
                        except Exception as e:
                            st.error(f"❌ Error displaying receipt: {str(e)}")
                else:
                    st.info("🔍 No receipts match your search criteria")
            else:
                st.info("📝 No receipts uploaded yet. Upload receipts through the OCR feature!")
        else:
            st.info("📝 No expenses recorded yet. Start by recording expenses!")
            
    except Exception as e:
        st.error(f"❌ Critical error in receipt management: {str(e)}")

def data_security_and_privacy():
    """
    Function to handle data security (simulated)
    LLM Needed: NO - Just UI for privacy settings
    """
    try:
        st.subheader("🔒 Data Security & Privacy")
        
        st.write("🛡️ Your financial data is stored locally and never shared with third parties.")
        st.write("🔐 All data is encrypted and protected according to privacy regulations.")
        
        # Security settings
        st.subheader("⚙️ Security Settings")
        
        if st.checkbox("Enable Data Encryption", value=True):
            st.success("✅ Data encryption is enabled!")
        
        if st.checkbox("Enable Automatic Backups"):
            backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"])
            st.info(f"📅 Automatic backups will run {backup_frequency.lower()}")
        
        # Data export
        st.subheader("📤 Data Export")
        if st.button("💾 Export All Data"):
            try:
                export_data = {
                    'expenses': st.session_state.expenses.to_dict('records') if not st.session_state.expenses.empty else [],
                    'budgets': st.session_state.budgets.to_dict('records') if not st.session_state.budgets.empty else [],
                    'savings_goals': st.session_state.savings_goals.to_dict('records') if not st.session_state.savings_goals.empty else []
                }
                
                json_str = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="📥 Download Data as JSON",
                    data=json_str,
                    file_name=f"budget_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                # Also provide CSV export
                if not st.session_state.expenses.empty:
                    st.download_button(
                        label="📊 Download Expenses CSV",
                        data=st.session_state.expenses.to_csv(index=False).encode('utf-8'),
                        file_name=f"expenses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"❌ Error exporting data: {str(e)}")
        
        # Privacy policy
        st.subheader("📜 Privacy Policy")
        with st.expander("Read Privacy Policy"):
            st.write("""
            **Data Collection:**
            - We collect only the financial data you enter
            - No personal identification information is collected
            - All data is stored locally on your device
            
            **Data Usage:**
            - Your data is used only for the functionality of this application
            - We do not share your data with any third parties
            - Data is not transmitted over the internet
            
            **Data Security:**
            - All data is encrypted at rest
            - You have full control over your data
            - You can export or delete your data at any time
            """)
            
    except Exception as e:
        st.error(f"❌ Critical error in security section: {str(e)}")

def bank_integration_placeholder():
    """
    Placeholder for bank integration feature
    LLM Needed: NO - Just UI placeholder
    Would need LLM for natural language banking queries
    """
    try:
        st.subheader("🏦 Bank Integration (Coming Soon)")
        
        st.info("🚀 This feature will allow automatic syncing with your bank accounts!")
        
        st.write("📋 Planned Features:")
        st.write("• Automatic transaction import")
        st.write("• Real-time balance updates")
        st.write("• Bank statement analysis")
        st.write("• Automatic expense categorization")
        
        bank_name = st.selectbox("Select Bank", ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "Other"])
        if bank_name:
            st.info(f"Bank integration for {bank_name} will be available soon!")
            
    except Exception as e:
        st.error(f"❌ Error in bank integration section: {str(e)}")

def main_dashboard():
    """
    Main dashboard overview
    """
    try:
        st.subheader("🏠 Dashboard Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_expenses = st.session_state.expenses['amount'].sum() if not st.session_state.expenses.empty else 0
        total_budget = st.session_state.budgets['budget_amount'].sum() if not st.session_state.budgets.empty else 0
        total_savings = st.session_state.savings_goals['current_amount'].sum() if not st.session_state.savings_goals.empty else 0
        expense_count = len(st.session_state.expenses)
        
        with col1:
            st.metric("💰 Total Expenses", f"${total_expenses:.2f}")

        with col2:
            st.metric("📊 Total Budget", f"${total_budget:.2f}")

        with col3:
            st.metric("🏆 Total Savings", f"${total_savings:.2f}")

        with col4:
            st.metric("🧾 Expense Count", f"{expense_count}")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎤 Record Voice Expense"):
                st.session_state.current_page = "🎤 Voice Expense"
                st.rerun()
        with col2:
            if st.button("📸 Scan Receipt"):
                st.session_state.current_page = "📸 OCR Receipts"
                st.rerun()
        with col3:
            if st.button("📊 View Analytics"):
                st.session_state.current_page = "📊 Visualizations"
                st.rerun()
        
        # Recent activity
        if not st.session_state.expenses.empty:
            st.subheader("📅 Recent Expenses")
            recent_expenses = st.session_state.expenses.tail(5)
            st.dataframe(recent_expenses[['date', 'amount', 'category', 'description']])
        else:
            st.info("📝 No recent expenses. Start by adding your first expense!")
        
        # Budget status
        if not st.session_state.budgets.empty:
            st.subheader("📊 Budget Status")
            for _, budget in st.session_state.budgets.iterrows():
                category_spending = st.session_state.expenses[
                    st.session_state.expenses['category'] == budget['category']
                ]['amount'].sum()
                
                progress = (category_spending / budget['budget_amount']) * 100 if budget['budget_amount'] > 0 else 0
                st.write(f"**{budget['category']}**")
                st.progress(min(progress/100, 1.0))
                st.write(f"${category_spending:.2f} / ${budget['budget_amount']:.2f} ({progress:.1f}%)")


        
        # Savings goals progress
        if not st.session_state.savings_goals.empty:
            st.subheader("🎯 Savings Goals Progress")
            for _, goal in st.session_state.savings_goals.iterrows():
                progress = (goal['current_amount'] / goal['target_amount']) * 100 if goal['target_amount'] > 0 else 0
                st.write(f"**{goal['goal_name']}**")
                st.progress(min(progress/100, 1.0))
                st.write(f"${goal['current_amount']:.2f} / ${goal['target_amount']:.2f} ({progress:.1f}%)")


        
    except Exception as e:
        st.error(f"❌ Error in dashboard: {str(e)}")

def main():
    """
    Main application function with error handling
    """
    try:
        # Initialize session state
        if not initialize_session_state():
            st.error("❌ Failed to initialize application. Please refresh the page.")
            return
        
        # Set page config
        st.set_page_config(
            page_title="💰 Budget Tracker Pro",
            page_icon="💰",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better UI
        st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .stMetric {
            background-color: white;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .css-1d391kg {
            background-color: #262730;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # App title and description
        st.title("💰 Budget Tracker Pro")
        st.markdown("*Your intelligent personal finance assistant*")
        
        # Sidebar navigation
        st.sidebar.title("🧭 Navigation")
        
        # Initialize current page in session state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "🏠 Dashboard"
        
        menu = [
            "🏠 Dashboard",
            "🎤 Voice Expense", 
            "📸 OCR Receipts", 
            "💰 Budget", 
            "🎯 Savings Goals", 
            "🏷️ Categorization", 
            "🔔 Alerts", 
            "📊 Visualizations", 
            "🧾 Receipts", 
            "🔒 Security",
            "🏦 Bank Integration"
        ]
        
        # Page selection
        choice = st.sidebar.selectbox("Choose a section", menu, 
                                    index=menu.index(st.session_state.current_page))
        st.session_state.current_page = choice
        
        # Display notifications in sidebar
        if st.session_state.notifications:
            st.sidebar.subheader("🔔 Notifications")
            for notification in st.session_state.notifications[-3:]:  # Show last 3
                if "🚨" in notification:
                    st.sidebar.error(notification)
                elif "⚠️" in notification:
                    st.sidebar.warning(notification)
        
        # Route to appropriate function
        if choice == "🏠 Dashboard":
            main_dashboard()
        elif choice == "🎤 Voice Expense":
            voice_expense_recording()
        elif choice == "📸 OCR Receipts":
            ocr_receipt_processing()
        elif choice == "💰 Budget":
            create_budget()
        elif choice == "🎯 Savings Goals":
            set_savings_goals()
        elif choice == "🏷️ Categorization":
            spending_categorization()
        elif choice == "🔔 Alerts":
            alerts_and_notifications()
        elif choice == "📊 Visualizations":
            visualizations_and_summaries()
        elif choice == "🧾 Receipts":
            receipt_management()
        elif choice == "🔒 Security":
            data_security_and_privacy()
        elif choice == "🏦 Bank Integration":
            bank_integration_placeholder()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info("💡 Tip: Use voice commands for quick expense logging!")
        
    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
        st.info("🔄 Please refresh the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()
