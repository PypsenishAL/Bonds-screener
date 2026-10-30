"""Streamlit entry point, with explicit local/published data mode."""
import os
import streamlit as st
from moex_bonds.ui.app import render_app
from moex_bonds.paths import PROJECT_DIR
from moex_bonds.publication.bundle import data_mode, validate_bundle

if __name__ == '__main__':
    settings = dict(os.environ)
    if 'BONDS_DATA_MODE' not in settings:
        try:
            settings['BONDS_DATA_MODE'] = st.secrets.get('BONDS_DATA_MODE', 'local')
        except FileNotFoundError:
            pass
    try:
        if data_mode(settings) == 'published':
            root = PROJECT_DIR / 'data/published'
            publication = validate_bundle(root)
            render_app(root/'spread_analysis', root/'daily_data', publication=publication)
        else:
            render_app()
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error('Published data unavailable: ' + str(error))
        st.stop()
