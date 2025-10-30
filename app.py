import streamlit as st
import json
import time
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import io
import cv2
import numpy as np
import hashlib 
import collections 

# --------------------------------------------------------------------------
# --- STEP 1: UNCOMMENTED FOR LOCAL RUNNING! 
# --------------------------------------------------------------------------
import av 
# Note: VideoTransformerBase is the base class for video processing
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode 

# --- 0. AUTHENTICATION & SESSION STATE SETUP ---

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

if 'users' not in st.session_state:
    st.session_state.users = {
        "admin": hashlib.sha256("password123".encode()).hexdigest(),
        "tester": hashlib.sha256("test".encode()).hexdigest()
    }
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = {} 

# Added state for displaying the *current* detection result in the main UI thread
if 'latest_detection_display' not in st.session_state:
    st.session_state.latest_detection_display = None


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def logout():
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.rerun() 

# --- 1. CONFIGURATION & RESOURCES ---

@st.cache_resource
def load_resources():
    st.markdown("---")
    try:
        # Load the AI Model (YOLOv8)
        # NOTE: 'model/best.pt' must be in a 'model' directory next to app.py 
        # This will load the model you just trained!
        model = YOLO('model/best.pt')
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure 'model/best.pt' exists.")
        model = None
        
    try:
        # Load mock rules for classification
        with open('assets/local_rules.json', 'r') as f:
            rules_db = json.load(f)
    except Exception:
        rules_db = {}
        
    try:
        # Load mock hubs for resource connection
        with open('assets/mock_hubs.json', 'r') as f:
            hubs_db = json.load(f)
    except Exception:
        hubs_db = {}
    
    try:
        # Load mock scan data for the heatmap
        with open('assets/scan_data.json', 'r') as f:
            scan_data = json.load(f)
    except Exception:
        scan_data = []

    return {"model": model, "rules": rules_db, "hubs": hubs_db, "scans": scan_data} 

# --- 2. CORE PREDICTION & LIVE TRANSFORMER CLASS ---

# --------------------------------------------------------------------------
# --- FIXED CLASS: Implements 'recv()' instead of the deprecated 'transform()' ---
# --------------------------------------------------------------------------
class YoloVideoTransformer(VideoTransformerBase):
    def __init__(self, model, result_buffer): 
        self.model = model
        self.rules_db = load_resources()['rules']
        self.result_buffer = result_buffer 

    # *** FIX 1: Renamed 'transform' to 'recv' for modern Streamlit-webrtc API ***
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the AV frame to an OpenCV image (NumPy array)
        img = frame.to_ndarray(format="bgr24")
        
        # Run YOLO detection
        results = self.model(img, stream=False, verbose=False)
        detected_item_tag = "No_Item"
        
        # Process results for plotting and determining classification
        for r in results:
            # r.plot() returns the image with bounding boxes drawn
            img = r.plot() 
            if len(r.boxes) > 0:
                # Get the class name of the highest confidence detection
                class_id = int(r.boxes[0].cls[0])
                detected_item_tag = self.model.names[class_id]
                break
        
        # Get rule and determine classification
        rule = self.rules_db.get(detected_item_tag, {"bin_type": "❓ UNKNOWN ITEM"})
        
        # Simple classification logic: Recycle/Compost = Usable
        if "Blue" in rule['bin_type'] or "Green" in rule['bin_type'] or "Brown" in rule['bin_type']:
            classification = "Usable (Recycle/Compost)"
        else:
            classification = "Useless (Landfill)"
        
        # Store information in the thread-safe buffer, NOT st.session_state!
        latest_info = {
            "item": detected_item_tag.replace('_', ' ').title(),
            "classification": classification,
            "time": time.strftime("%H:%M:%S"),
            "bin_type": rule['bin_type']
        }
        
        # Use deque to safely pass the latest result to the main thread
        # maxlen=1 ensures we only keep the newest result, avoiding data overload
        if len(self.result_buffer) > 0:
            self.result_buffer.popleft()
        self.result_buffer.append(latest_info)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
# --------------------------------------------------------------------------

def predict_and_get_rule(uploaded_file, resources):
    
    model = resources['model']
    rules_db = resources['rules']
    hubs_db = resources['hubs']

    if model is None:
        return "MODEL_ERROR", None, None, None

    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    results = model(image, stream=False)

    detected_item_tag = "NOT_DETECTED"
    annotated_image_np = np.array(image)
    
    for r in results:
        if len(r.boxes) > 0:
            top_box = r.boxes[0]
            class_id = int(top_box.cls[0])
            detected_item_tag = model.names[class_id]
            annotated_image_np = r.plot() 
            break

    annotated_image = Image.fromarray(cv2.cvtColor(annotated_image_np, cv2.COLOR_BGR2RGB))
    
    default_rule = {
        "bin_type": "❓ UNKNOWN ITEM",
        "instruction": "This item is recognized by the AI but not yet in the Local Rules Engine. Please check municipal website.",
        "hub_id": None
    }
    rule = rules_db.get(detected_item_tag, default_rule)
    
    hub_id = rule.get('hub_id')
    hub = hubs_db.get(hub_id, {"name": "No specialized hub connected", "link": "#", "address": "N/A"})
    
    return detected_item_tag, rule, hub, annotated_image

# --- 3. HEATMAP PAGE FUNCTION ---

def show_heatmap_page(resources):
    st.title("📈 Waste-to-Resource Heatmap Analysis")
    st.subheader("Visualizing material density for collection optimization and investment.")
    
    scan_data = resources['scans']
    if not scan_data:
        st.warning("No mock scan data available to generate the map. Check assets/scan_data.json.")
        return

    df = pd.DataFrame(scan_data)
    df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(1)
    
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.error("Mock scan data is missing 'latitude' or 'longitude' columns.")
        return

    st.markdown("---")
    item_options = ['All Items'] + df['item'].unique().tolist()
    selected_item = st.selectbox("Filter Material Hotspots:", item_options)

    if selected_item != 'All Items':
        filtered_df = df[df['item'] == selected_item]
        st.info(f"Showing demand hotspots for **{selected_item.title().replace('-', ' ')}** only.")
    else:
        filtered_df = df
        st.info("Showing combined material density.")

    if not filtered_df.empty:
        st.markdown("### 🗺️ Material Concentration Map")
        map_data = filtered_df[['latitude', 'longitude']].copy()
        st.map(map_data, zoom=11)
        
        st.markdown("### 📊 Raw Data Insights (Mock Scans)")
        # FIX: Replaced use_container_width=True with width='stretch'
        st.dataframe(filtered_df[['item', 'time', 'latitude', 'longitude', 'value']], width='stretch')
        
        st.markdown("""
        <div style="padding: 10px; border-left: 5px solid #00AA77; background-color: #e0fff0;">
        **Pitch Point:** This feature demonstrates **Business Intelligence (BI)**. We turn collective user actions 
        into real-time data for optimizing collection logistics.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"No scan data found for the selected material.")

# --- 4. LIVE CAMERA & FEEDBACK PAGE ---

# Mock function to simulate a detection event for the history table 
# This runs in the sandboxed environment to demonstrate the logging feature.
def mock_live_detect(user, rules_db):
    # Updated items to reflect the 6 trained classes
    items = ["Apple", "Carrot", "Glass Bottle", "Glass-Plate", "Paper", "Plastic-Bottle"]
    random_item = np.random.choice(items)
    
    rule = rules_db.get(random_item, {"bin_type": "Black Bin (Landfill)"})
    
    if "Black" in rule['bin_type'] or "UNKNOWN" in rule['bin_type']:
        classification = "Useless (Landfill)"
    else:
        classification = "Usable (Recycle/Compost)"
    
    new_entry = {
        "item": random_item.replace('_', ' ').title(),
        "time": time.strftime("%H:%M:%S"),
        "bin_type": rule['bin_type'],
        "classification": classification
    }
    
    if user not in st.session_state.detection_history:
        st.session_state.detection_history[user] = []
        
    st.session_state.detection_history[user].insert(0, new_entry) 
    
    return new_entry

def show_live_scanner_page(resources):
    st.title("📹 Live Waste Detection & Feedback Loop")
    st.subheader("Real-time sorting and generating training data.")
    
    user = st.session_state.current_user
    
    col1, col2 = st.columns([2, 1])
    
    # Initialize the thread-safe buffer ONLY here, outside the transformer
    # We use a deque with maxlen=1 to always hold the latest detection result
    if 'detection_buffer' not in st.session_state:
        st.session_state.detection_buffer = collections.deque(maxlen=1)
        
    detection_buffer = st.session_state.detection_buffer # Get reference to the buffer
    
    with col1:
        st.markdown("### 🤖 Live AI Stream")
        
        # Use an empty placeholder to update the latest detection result dynamically
        detection_placeholder = st.empty()
        
        # --------------------------------------------------------------------------
        # --- FIX 3: Updated 'webrtc_streamer' arguments ---
        # --------------------------------------------------------------------------
        webrtc_streamer(
            key="yolo-stream",
            mode=WebRtcMode.SENDRECV,
            # FIX 3a: Renamed video_transformer_factory to video_processor_factory
            video_processor_factory=lambda: YoloVideoTransformer(resources['model'], detection_buffer),
            # FIX 3b: Renamed async_transform to async_processing
            async_processing=True, 
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        # --------------------------------------------------------------------------
        
        # --- FIX: LOGIC TO READ THE BUFFER AND UPDATE STATE ---
        
        # Check the buffer for new results only if the buffer is not empty
        if len(detection_buffer) > 0:
            latest_info = detection_buffer[0]
            
            # Update the latest display state
            st.session_state.latest_detection_display = latest_info

            # Log to history if the current detection is significantly different or after a delay
            # Note: For simplicity here, we rely on Streamlit's implicit rerun to update
            
            # Log to history if it's a new result not yet in the history
            history = st.session_state.detection_history.get(user, [])
            if not history or latest_info != history[0]:
                 if user not in st.session_state.detection_history: st.session_state.detection_history[user] = []
                 st.session_state.detection_history[user].insert(0, latest_info)
                 
        # Display the latest detection result in the placeholder
        if st.session_state.latest_detection_display:
            latest_detection = st.session_state.latest_detection_display
            
            item_display = latest_detection['item']
            bin_type_display = latest_detection['bin_type']
            time_display = latest_detection['time']
            classification_display = latest_detection['classification']

            if "Usable" in classification_display:
                detection_placeholder.success(f"✅ DETECTED: **{item_display}** | Disposal: **{bin_type_display}** | Classification: **{classification_display}** | Time: {time_display}")
            else:
                detection_placeholder.error(f"❌ DETECTED: **{item_display}** | Disposal: **{bin_type_display}** | Classification: **{classification_display}** | Time: {time_display}")
        else:
            detection_placeholder.info("Waiting for live detection data...")

        st.markdown("---")
        # This button simulates the detection result for testing the history table logic
        if st.button("Simulate Live Detection & Log to History"):
            with st.spinner("Processing simulated frame..."):
                latest_detection = mock_live_detect(user, resources['rules'])
                
                # Update the latest display state from simulation
                st.session_state.latest_detection_display = latest_detection
                
                # Display the simulated result card
                st.markdown("---")
                if "Usable" in latest_detection['classification']:
                    st.success(f"✅ DETECTED: {latest_detection['item']} -> {latest_detection['classification']}")
                else:
                    st.error(f"❌ DETECTED: {latest_detection['item']} -> {latest_detection['classification']}")
                st.markdown(f"**Disposal:** {latest_detection['bin_type']} | **Time:** {latest_detection['time']}")
        
    with col2:
        st.markdown(f"### 📊 Your Detection History ({user})")
        st.caption("Data points collected for model refinement.")
        
        history = st.session_state.detection_history.get(user, [])
        
        if not history:
            st.info("No live detection events recorded yet. Simulate one!")
        else:
            history_df = pd.DataFrame(history)
            
            def color_classification(val):
                color = 'lightgreen' if 'Usable' in val else 'salmon'
                return f'background-color: {color}'

            # FIX 4: Changed applymap to map, and use_container_width=True to width='stretch'
            st.dataframe(history_df.style.map(color_classification, subset=['classification']), width='stretch')
            
            st.markdown("""
            <div style="padding: 10px; border-left: 5px solid #FF8C00; background-color: #fff8e1;">
            **Pitch Point:** This table is our **Self-Refinement Data**. Every logged item 
            improves the Usable vs. Useless separation accuracy.
            </div>
            """, unsafe_allow_html=True)


# --- 5. DASHBOARD (Combined Scanner & Heatmap & Live) ---

def show_dashboard(resources):
    
    st.sidebar.title("🛠️ CircuScan Dashboard")
    st.sidebar.markdown(f"**Welcome back, {st.session_state.current_user}!**")
    st.sidebar.button("Logout", on_click=logout)
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Navigation", 
                             ["1. Upload Scanner", 
                              "2. Live Camera/Feedback",
                              "3. Resource Heatmap"])

    st.sidebar.markdown("### Status")
    if resources['model'] is None:
        st.sidebar.error("🚨 AI Model Failed to Load.")
    else:
        st.sidebar.success("🤖 AI Model Loaded.")
    st.sidebar.caption("All resources online.")
    st.sidebar.markdown("---")

    if page == "1. Upload Scanner":
        
        st.title("♻️ CircuScan: AI Waste Segregation")
        st.subheader("Your fast-track localized recycling and circular economy advisor.")
        
        st.sidebar.markdown("### Scanner Settings")
        city = st.sidebar.selectbox(
            "Select Local Municipality (Mock Rules):",
            ("New York City (Mock Rules)", "San Francisco (Mock Rules)", "General Rules (Default)")
        )
        st.sidebar.info(f"Active Ruleset: **{city}**")
        
        st.sidebar.slider(
            "AI Confidence Threshold:",
            min_value=0.0, max_value=1.0, value=0.65, step=0.05
        )
        
        st.markdown("---")
        
        st.subheader(f"1️⃣ Upload Item for Scan (Rules: {city})")
        uploaded_file = st.file_uploader("Choose an image of a waste item:", type=['jpg', 'jpeg', 'png'])
        
        st.markdown("---")

        if uploaded_file is not None:
            
            st.subheader("Processing...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("1/3: Sending image to AI model...")
            progress_bar.progress(33)
            time.sleep(0.5)

            status_text.text("2/3: Running YOLO inference and classification...")
            progress_bar.progress(66)
            time.sleep(0.5)

            status_text.text("3/3: Consulting Local Rules Engine...")
            progress_bar.progress(99)
            time.sleep(0.2)
            
            tag, rule, hub, annotated_image = predict_and_get_rule(uploaded_file, resources)
            
            progress_bar.progress(100)
            status_text.success("Analysis Complete!")
            time.sleep(0.5)
            st.empty() 

            st.subheader("2️⃣ Analysis Results")
            
            if tag in ["MODEL_ERROR", "NOT_DETECTED"]:
                if tag == "MODEL_ERROR":
                    st.error("🚨 System Error: The AI model failed to load. Cannot proceed.")
                elif tag == "NOT_DETECTED":
                    st.warning("⚠️ No clear waste item was detected in the image. Please try a clearer picture.")
                return

            st.markdown(f"### Item Classified as: **{tag.replace('_', ' ').title()}**")
            
            col_img, col_instructions, col_hub = st.columns([1.5, 2, 1.5])

            with col_img:
                # FIX: Replaced use_container_width=True with width='stretch'
                st.image(annotated_image, caption='AI Detection', width='stretch')

            with col_instructions:
                st.markdown("### 🗑️ Local Disposal Instructions")
                
                if "Black" in rule['bin_type'] or "UNKNOWN" in rule['bin_type']:
                    st_alert = st.error
                    icon = "🔥" 
                elif "Brown" in rule['bin_type'] or "Compost" in rule['bin_type']:
                    st_alert = st.info
                    icon = "🌳" 
                else:
                    st_alert = st.success
                    icon = "💎" 
                
                st_alert(f"**{icon} BIN TYPE for {city}:** {rule['bin_type']}")
                st.markdown(f"**Action Required:** {rule['instruction']}")

            with col_hub:
                st.markdown("### 📍 Circular Hub Connection")
                st.caption("Extending the life cycle beyond the bin.")

                if hub['name'] != "No specialized hub connected":
                    st.markdown(f"""
                    <div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f0f2f6;">
                        ♻️ **{hub['category']}**
                        <p style='margin-bottom: 5px;'>**Hub:** <a href='{hub['link']}' target='_blank'>{hub['name']}</a></p>
                        <small>Address: {hub.get('address', 'N/A')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No specialized local hub link found for this item.")
                    
    elif page == "2. Live Camera/Feedback":
        show_live_scanner_page(resources)

    elif page == "3. Resource Heatmap":
        show_heatmap_page(resources)


# --- 6. AUTHENTICATION UI PAGES (UNCHANGED) ---

def login_page():
    st.title("🔒 Login to CircuScan")
    st.subheader("Personalized recycling rules and dashboard analytics.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username in st.session_state.users:
                hashed_pw = hash_password(password)
                if st.session_state.users[username] == hashed_pw:
                    st.session_state.logged_in = True
                    st.session_state.current_user = username
                    st.success(f"Welcome, {username}! Redirecting...")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid Username or Password.")
            else:
                st.error("Invalid Username or Password.")

    st.markdown("---")
    if st.button("Need an account? Sign Up"):
        st.session_state.auth_mode = "signup"
        st.rerun()
        
def signup_page():
    st.title("📝 Sign Up for CircuScan")
    st.subheader("Start making a difference with personalized guidance.")

    with st.form("signup_form"):
        new_username = st.text_input("Choose Username (e.g., JaneDoe)")
        new_password = st.text_input("Choose Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create Account")

        if submitted:
            if not new_username or not new_password:
                st.error("Username and Password cannot be empty.")
            elif new_username in st.session_state.users:
                st.error("This username already exists.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                st.session_state.users[new_username] = hash_password(new_password)
                st.success(f"Account for {new_username} created successfully! Please log in.")
                st.session_state.auth_mode = "login"
                st.rerun()

    st.markdown("---")
    if st.button("Already have an account? Login"):
        st.session_state.auth_mode = "login"
        st.rerun()

# --- 7. MAIN ROUTER ---

def main():
    st.set_page_config(page_title="CircuScan AI Segregation", layout="wide")
    resources = load_resources()

    if st.session_state.logged_in:
        show_dashboard(resources)
    else:
        st.sidebar.title("CircuScan Access")
        st.sidebar.markdown("---")
        
        if 'auth_mode' not in st.session_state:
            st.session_state.auth_mode = "login"
        
        if st.session_state.auth_mode == "login":
            login_page()
        else:
            signup_page()

if __name__ == "__main__":
    main()
