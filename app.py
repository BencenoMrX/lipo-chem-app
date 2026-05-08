import streamlit as st
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
import py3Dmol
from stmol import showmol
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- Helper Function for PubChem API ---
def get_smiles_from_name(name):
    """Queries the PubChem API to translate a common name into a SMILES string."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
        return None
    except:
        return None

# Set up the page layout
st.set_page_config(page_title="Flavor Chemistry Viewer", layout="wide")
st.title("Flavor & Pigment Molecule Viewer")

# Updated user input to accept both
user_input = st.text_input("Enter a Chemical Name (e.g., Limonene) or SMILES:", "Vanillin")

if user_input:
    # 1. First, assume the input is a SMILES string
    mol = Chem.MolFromSmiles(user_input)
    smiles_to_render = user_input
    
    # 2. If RDKit fails (meaning it's a word, not a SMILES), ask PubChem
    if mol is None:
        with st.spinner(f"Looking up '{user_input}' in PubChem..."):
            fetched_smiles = get_smiles_from_name(user_input)
            
            if fetched_smiles:
                st.success(f"Found structure for {user_input}: {fetched_smiles}")
                mol = Chem.MolFromSmiles(fetched_smiles)
                smiles_to_render = fetched_smiles
            else:
                st.error(f"Could not find a structure for '{user_input}'. Please check the spelling.")

    # 3. If we successfully built a molecule object, render the UI
    if mol is not None:
        # --- PART 1: 3D Generation & Calculations ---
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_3d)
        
        # Calculate Properties
        tpsa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
        volume = AllChem.ComputeMolVolume(mol_3d)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        
        # --- PART 2: UI Layout ---
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Molecular Properties")
            st.metric("LogP (Lipophilicity)", f"{logp:.2f}")
            st.metric("TPSA (Polar Surface Area)", f"{tpsa:.2f} Å²")
            st.metric("Volume", f"{volume:.2f} Å³")
            st.metric("H-Bond Acceptors", hba)
            st.metric("H-Bond Donors", hbd)
            st.info("Note: TPSA, Volume, and H-bonding sites dictate a molecule's polarity and its ability to form intermolecular interactions.")
            
        with col2:
            st.subheader("3D Lipophilicity Map")
            # Calculate atomic contributions to logP
            contribs = [x[0] for x in rdMolDescriptors._CalcCrippenContribs(mol_3d)]
            norm = mcolors.Normalize(vmin=min(contribs), vmax=max(contribs))
            cmap = cm.coolwarm
            
            # Map colors to atoms
            atom_colors = {i: mcolors.to_hex(cmap(norm(contrib))) for i, contrib in enumerate(contribs)}
            
            # Generate py3Dmol view
            mb = Chem.MolToMolBlock(mol_3d)
            view = py3Dmol.view(width=600, height=500)
            view.addModel(mb, 'sdf')
            
            for i, color in atom_colors.items():
                view.setStyle({'index': i}, {
                    'stick': {'color': color, 'radius': 0.15}, 
                    'sphere': {'color': color, 'radius': 0.3}
                })
                
            view.addSurface(py3Dmol.VDW, {'opacity': 0.6})
            view.zoomTo()
            
            # Render in Streamlit
            showmol(view, height=500, width=600)
