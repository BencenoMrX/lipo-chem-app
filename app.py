import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
import py3Dmol
from stmol import showmol
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Set up the page layout
st.set_page_config(page_title="Flavor Chemistry Viewer", layout="wide")
st.title("Flavor & Pigment Molecule Viewer")

# User input for SMILES
smiles = st.text_input("Enter a SMILES string:", "O=Cc1ccc(O)c(OC)c1")

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    
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
        col1, col2 = st.columns([1, 2]) # Makes the right column twice as wide
        
        with col1:
            st.subheader("Molecular Properties")
            st.metric("LogP (Lipophilicity)", f"{logp:.2f}")
            st.metric("TPSA (Polar Surface Area)", f"{tpsa:.2f} Å²")
            st.metric("Volume", f"{volume:.2f} Å³")
            st.metric("H-Bond Acceptors", hba)
            st.metric("H-Bond Donors", hbd)
            st.info("Note: TPSA, Volume, and H-bonding sites dictate a molecule's polarity and its ability to form intermolecular interactions with solvents or olfactory receptors.")
            
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
            
            # Render in Streamlit using the stmol wrapper
            showmol(view, height=500, width=600)
            
    else:
        st.error("Invalid SMILES string. Please check your input.")