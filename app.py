import streamlit as st
from PIL import Image
import json
import vision

st.set_page_config(page_title="Automated Inventory & Nutritional Intelligence", layout="wide", page_icon="🍎")

def load_database():
    try:
        with open("local_db.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

db = load_database()
ai_model = vision.load_vgg_model()

st.title("🍎 Automated Inventory & Nutritional Intelligence")
st.divider()

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📸 Scan Item")
    img_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])
    img = None

    if img_file is not None:
        img = Image.open(img_file)
        if img.mode != "RGB":
            img = img.convert("RGB")
        st.image(img, use_container_width=True)

with col2:
    st.subheader("🔍 Intelligence Report")
    
    if img:
        with st.spinner("Analyzing image..."):
            prediction, category, confidence = vision.classify_image(img, ai_model)
            
        st.success(f"**Detected Item:** {prediction}")
        
        c1, c2 = st.columns(2)
        c1.metric("Classification", category)
        c2.metric("AI Confidence", f"{confidence * 100:.1f}%")
        
        st.divider()
        
        search_term = prediction.lower()
        item_data = db.get(search_term)

        if item_data is not None:
            st.markdown("#### ⏳ Storage & Shelf-Life")
            st.warning(f"**Consume within:** {item_data['days']} Days")
            st.info(f"**Optimal Storage:** {item_data['tip']}")
            
            st.markdown("#### 🥗 Nutritional Profile (Per 100g)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Calories", f"{item_data['calories']} kcal")
            m2.metric("Protein", f"{item_data['protein']} g")
            m3.metric("Carbs", f"{item_data['carbs']} g")
            
        else:
            st.error(f"⚠️ No database entry found for '{prediction}'.")
    else:
        st.info("👈 Upload an image to generate the intelligence report.")