"""Streamlit entry point, with explicit local/published data mode."""
import os
import streamlit as st
from moex_bonds.ui.app import render_app
from moex_bonds.paths import PROJECT_DIR
from moex_bonds.publication.bundle import data_mode, validate_bundle
from moex_bonds.ui.app import file_stamp


@st.cache_data(max_entries=2,show_spinner=False)
def published_bundle(root, stamp):
    # Validate every byte on a changed publication, reuse validation between clicks.
    return validate_bundle(root)

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
            stamp = tuple((str(p.relative_to(root)),file_stamp(p)) for p in sorted(root.rglob('*')) if p.is_file())
            publication = published_bundle(str(root),stamp)
            render_app(root/'spread_analysis', root/'daily_data', publication=publication)
        else:
            render_app()
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error('Published data unavailable: ' + str(error))
        st.stop()
