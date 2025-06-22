import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# VAE Model Definition (same as training script)
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.encoder_fc1(x))
        h2 = F.relu(self.encoder_fc2(h1))
        mu = self.mu_layer(h2)
        logvar = self.logvar_layer(h2)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h1 = F.relu(self.decoder_fc1(z))
        h2 = F.relu(self.decoder_fc2(h1))
        return torch.sigmoid(self.decoder_output(h2))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

@st.cache_resource
def load_model():
    """Load the trained VAE model"""
    try:
        # For deployment, you'll need to upload the model file
        checkpoint = torch.load('mnist_vae_model.pth', map_location='cpu')
        
        model_config = checkpoint['model_config']
        model = VAE(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            latent_dim=model_config['latent_dim']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        digit_stats = checkpoint['digit_stats']
        
        return model, digit_stats
    
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'mnist_vae_model.pth' is uploaded.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_digit_images(model, digit, num_samples=5, digit_stats=None, latent_dim=20):
    """Generate images for a specific digit"""
    model.eval()
    with torch.no_grad():
        if digit_stats and digit in digit_stats:
            # Use digit-specific statistics for better generation
            mean = torch.tensor(digit_stats[digit]['mean'])
            std = torch.tensor(digit_stats[digit]['std'])
            # Add some randomness to create variety
            z = torch.randn(num_samples, latent_dim) * (std * 0.8) + mean + torch.randn(num_samples, latent_dim) * 0.2
        else:
            # Fallback to random sampling
            z = torch.randn(num_samples, latent_dim)
        
        samples = model.decode(z)
        return samples.view(num_samples, 28, 28).numpy()

def create_image_grid(images, titles=None):
    """Create a grid of images"""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
        if titles:
            axes[i].set_title(titles[i], fontsize=12)
        else:
            axes[i].set_title(f'Sample {i+1}', fontsize=12)
    
    plt.tight_layout()
    return fig

# Main Streamlit App
def main():
    # Header
    st.title("🔢 Handwritten Digit Image Generator")
    st.markdown("Generate synthetic MNIST-like images using a trained Variational Autoencoder")
    
    # Load model
    model, digit_stats = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar for controls
    st.sidebar.header("Generation Controls")
    
    # Digit selection
    selected_digit = st.sidebar.selectbox(
        "Choose a digit to generate (0-9):",
        options=list(range(10)),
        index=2
    )
    
    # Generation button
    if st.sidebar.button("🎲 Generate Images", type="primary"):
        with st.spinner("Generating handwritten digits..."):
            # Generate images
            generated_images = generate_digit_images(
                model, 
                selected_digit, 
                num_samples=5, 
                digit_stats=digit_stats
            )
            
            # Store in session state
            st.session_state.generated_images = generated_images
            st.session_state.current_digit = selected_digit
    
    # Display results
    if 'generated_images' in st.session_state:
        st.header(f"Generated Images of Digit {st.session_state.current_digit}")
        
        # Create and display the image grid
        fig = create_image_grid(
            st.session_state.generated_images,
            titles=[f"Sample {i+1}" for i in range(5)]
        )
        
        st.pyplot(fig)
        
        # Display individual images in columns
        st.subheader("Individual Samples")
        cols = st.columns(5)
        
        for i, (col, img) in enumerate(zip(cols, st.session_state.generated_images)):
            with col:
                # Convert to PIL Image for better display
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                img_pil = img_pil.resize((112, 112), Image.NEAREST)  # Upscale for better visibility
                st.image(img_pil, caption=f"Sample {i+1}", use_column_width=True)
    
    else:
        # Default display
        st.info("👆 Select a digit and click 'Generate Images' to create handwritten digit samples!")
        
        # Show example of what the app can do
        st.subheader("About This App")
        st.markdown("""
        This web application uses a **Variational Autoencoder (VAE)** trained on the MNIST dataset to generate 
        synthetic handwritten digits. Here's what it does:
        
        - 🎯 **Targeted Generation**: Select any digit (0-9) to generate
        - 🎨 **Multiple Samples**: Creates 5 different variations of the selected digit
        - 🧠 **Custom Trained Model**: Uses a VAE model trained from scratch (not pre-trained)
        - 📊 **MNIST-like Output**: Generates 28x28 grayscale images similar to the MNIST dataset
        
        ### How it Works:
        1. The VAE learns the underlying distribution of handwritten digits
        2. It encodes digit-specific patterns into a latent space
        3. When generating, it samples from the learned distribution for the selected digit
        4. The decoder reconstructs these samples into recognizable digit images
        
        ### Model Details:
        - **Architecture**: Variational Autoencoder with 400 hidden units
        - **Latent Dimension**: 20-dimensional latent space
        - **Training**: Trained on full MNIST dataset using Google Colab T4 GPU
        - **Loss Function**: Combination of reconstruction loss (BCE) and KL divergence
        """)

if __name__ == "__main__":
    main()