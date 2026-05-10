import streamlit as st
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
import py3Dmol
from stmol import showmol
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- Helper Functions ---
def fetch_smiles(name):
    """Queries databases for the English chemical name."""
    try:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{name}/smiles"
        response = requests.get(url, timeout=5)
        if response.status_code == 200 and "<h1>" not in response.text:
            return response.text.strip()
    except:
        pass
        
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except:
        pass
    return None

def get_smiles_from_name(name):
    """Main pipeline: tries direct search, then tries Spanish-to-English translation."""
    smiles = fetch_smiles(name)
    if smiles:
        return smiles, name 
        
    try:
        wiki_url = "https://es.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "langlinks",
            "lllang": "en",
            "titles": name.capitalize(), 
            "format": "json"
        }
        res = requests.get(wiki_url, params=params, timeout=5).json()
        pages = res.get("query", {}).get("pages", {})
        
        for page_id, page_data in pages.items():
            if "langlinks" in page_data:
                english_name = page_data["langlinks"][0]["*"]
                smiles = fetch_smiles(english_name)
                if smiles:
                    return smiles, english_name 
    except:
        pass
    return None, None

# --- Main App ---
st.set_page_config(page_title="Chemistry Viewer", layout="wide")
st.title("Molecule 3D Viewer")

user_input = st.text_input("Enter a chemical name (e.g., limonene, agua) or SMILES:", "Vanillin")

if user_input:
    mol = Chem.MolFromSmiles(user_input)
    smiles_to_render = user_input
    
    if mol is None:
        with st.spinner(f"Searching databases for '{user_input}'..."):
            fetched_smiles, translated_name = get_smiles_from_name(user_input)
            
            if fetched_smiles:
                if translated_name and translated_name.lower() != user_input.lower():
                    st.success(f"Translated '{user_input}' to '{translated_name}'. Found structure: {fetched_smiles}")
                else:
                    st.success(f"Found structure for {user_input}: {fetched_smiles}")
                
                mol = Chem.MolFromSmiles(fetched_smiles)
                smiles_to_render = fetched_smiles
            else:
                st.error(f"Could not find a structure for '{user_input}'. Please check the spelling.")

    if mol is not None:
        # 1. 3D Generation
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_3d)
        
        # 2. Calculate Advanced Properties
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
        volume = AllChem.ComputeMolVolume(mol_3d)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        fcsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        
        # 3. UI Layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Physicochemical Properties")
            st.metric("Molecular Weight", f"{mw:.2f} g/mol")
            st.metric("LogP (Lipophilicity)", f"{logp:.2f}")
            st.metric("TPSA (Polar Surface Area)", f"{tpsa:.2f} Å²")
            st.metric("Volume", f"{volume:.2f} Å³")
            
            st.subheader("Intermolecular Interaction Drivers")
            st.metric("H-Bond Acceptors / Donors", f"{hba} / {hbd}")
            st.metric("Aromatic Rings (\u03C0-\u03C0 stacking)", arom_rings)
            #st.metric("Rotatable Bonds (Flexibility)", rot_bonds)
            st.metric("Fraction Csp3 (3D Character)", f"{fcsp3:.2f}")

            st.info("LogP calculated by Wildman-Crippen method, assigning a lipophilicity value to each atom depending on its chemical environment, and adding them over the molecule.")
            #st.info("Note: Properties like flexibility, volume, and aromaticity dictate if a flavor molecule can successfully bind to receptors or encapsulate within host matrices.")
            
        with col2:
            st.subheader("3D Molecular Map")
            
            # Interactive Controls for the Map
            control_col1, control_col2, control_col3 = st.columns(3)
            with control_col1:
                show_surface = st.checkbox("Show 3D surface", value=True)
                show_lipo = st.checkbox("Show atom lipophilicity", value=True)
            with control_col2:
                cmap_name = st.selectbox("Atom color scale", ["coolwarm", "PiYG", "viridis", "RdYlGn"])
            with control_col3:
                surf_type = st.selectbox("3D surface type", ["van der Waals", "Solvent accessible surface", "Solvent excluded surface"])

            surface_mapping = {
                    "van der Waals": py3Dmol.VDW,
                    #"MS": py3Dmol.MS,  
                    "Solvent accessible surface": py3Dmol.SAS,
                    "Solvent excluded surface": py3Dmol.SES 
                }
            
            # Setup py3Dmol
            mb = Chem.MolToMolBlock(mol_3d)
            view = py3Dmol.view(width=600, height=500)
            view.addModel(mb, 'sdf')
            
            # Calculate logP contributions and map the colormap
            contribs = [x[0] for x in rdMolDescriptors._CalcCrippenContribs(mol_3d)]
            cmap = cm.get_cmap(cmap_name)
            norm = mcolors.Normalize(vmin=min(contribs), vmax=max(contribs))

            # 1. Create a dictionary mapping the atom index to its new hex color
            color_map = {i: mcolors.to_hex(cmap(norm(contrib))) for i, contrib in enumerate(contribs)}

            # 2. Style the Sticks and Spheres
            if show_lipo:
                for i, color in color_map.items():
                    view.setStyle({'index': i}, {
                        'stick': {'color': color, 'radius': 0.15}, 
                        'sphere': {'color': color, 'radius': 0.3}
                    })
                
                # Draw the legend/colorbar using Matplotlib
                fig, ax = plt.subplots(figsize=(6, 0.4))
                fig.subplots_adjust(bottom=0.6)
                cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
                cb.set_label('More hydrophilic atoms   ←        →    More lipophilic atoms')
                st.pyplot(fig)
            else:
                # Default CPK coloring if the map is turned off
                view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'radius': 0.3}})

            # 3. Add the Surface
            if show_surface:
                selected_surface = surface_mapping[surf_type]
                surf_options = {'opacity': 0.7}
                
                # If lipophilicity is ON, force the surface to use our color_map via the atom index
                if show_lipo:
                    surf_options['colorscheme'] = {
                        'prop': 'index',
                        'map': color_map
                    }
                    
                view.addSurface(selected_surface, surf_options)

            view.zoomTo()
            showmol(view, height=600, width=600)
