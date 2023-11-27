from os import environ
import io
from html2docx import html2docx
from pdfkit import from_string
from markdown import markdown
import streamlit as st

from myfunc.mojafunkcija import st_style, positive_login, init_cond_llm
from custom_llm_agent import our_custom_agent

version = "16.11.23. Dj OpenAI"

st_style()

def main():
    default_session_states = {
        "model": "",
        "temp": 1.0,
        "alpha": 0.5,
        "text": "text",
        "index_name": "positive",
        "namespace": "pravnik",
        "odgovor": "",
        "tematika": "",
        "broj_k": 5,
        "stil": "",
        "score": 0.1,
        "sql_base_name": "test1"
        }
    st.session_state = {**default_session_states, **st.session_state}

    st.markdown(
        f"<p style='font-size: 10px; color: grey;'>{version}</p>",
        unsafe_allow_html=True,
        )
    st.subheader("Test App za custom agenta 🏙️")

    with st.expander("Pročitajte uputstvo 🧝"):
        st.caption(
            body="""
            Hybrid search se bazira na dvostrukoj pretrazi indeksa: prema kjucnim recima i prema semantickom znacenju.
            App ce kombinovati odgovre na obe pretrage i predloziti podatke potrebne za odgovor. 
            Odatle ce LLM preuzeti zadatak da odgovori na pitanje. Trenutno nisu podrzani namsepace-ovi.
            """)

    with st.sidebar:
        st.session_state["model"], st.session_state["temp"] = init_cond_llm()

        st.session_state["stil"] = (
            "You are a helpful assistent. You always answer in the Serbian language."
            )
        st.caption(
            body="Temperatura za hybrid search treba de je što bliže 0"
            )
        st.session_state["broj_k"] = st.number_input(
            label="Set number of returned documents",
            min_value=1, 
            max_value=10,
            value=5, 
            step=1,
            key="broj_k_key",
            help="Broj dokumenata koji se vraćaju iz indeksa", 
            )
        st.session_state["score"] = st.slider(
            label="Set score",
            min_value=0.00, 
            max_value=2.00, 
            value=0.10, 
            step=0.01,
            help="Koeficijent koji određuje kolji će biti prag relevantnosti dokumenata uzetih u obzir za odgovore. \
                0 je svi dokumenti, veci broj je stroziji kriterijum. Score u hybrid searchu moze biti proizvoljno veliki.",
            )
        st.session_state["namespace"] = st.selectbox(
            label="Odaberite oblast",
            options=(
                "pravnik",
                "positive",
                "zapisnici",
                ),
            )
        st.session_state["sql_base_name"] = st.text_input(
            label="Unesite naziv SQL baze", value="test1", key="sql_baza")

    zahtev = ""
    prompt_file = st.file_uploader(
        label="Izaberite početni prompt koji možete editovati ili pišite prompt od početka za definisanje vašeg zahteva",
        key="upload_prompt",
        type="txt",
        )
     
    with st.form(key="stilovi", clear_on_submit=False):
        zahtev = st.text_area(
            label="Opišite temu, iz oblasti Positive, ili opšte teme. Objasnite i formu željenog teksta: ",
            value=prompt_file.getvalue().decode("utf-8") if prompt_file is not None else " ",
            key="prompt_prva",
            height=150,
            )
        st.form_submit_button(label="Submit")


    if zahtev not in ["", " "]:
        st.session_state["odgovor"] = our_custom_agent(zahtev, dict(st.session_state))
        with st.spinner("Sačekajte trenutak..."):
            try:
                # st.session_state["odgovor"] = our_custom_agent(zahtev, dict(st.session_state))
                st.write(st.session_state["odgovor"])
            except Exception as e:
                st.warning(f"Nisam u mogućnosti da završim tekst. Ovo je opis greške:\n\n {e}")


    if st.session_state["odgovor"] != "":
        with st.expander("FINALNI TEKST", expanded=True):
            st.markdown(body=st.session_state["odgovor"])
        html = markdown(text=st.session_state["odgovor"])

        try:
            pdf_data = from_string(input=html, cover_first=False, 
                                   options={"encoding": "UTF-8", "no-outline": None, "quiet": "",},
                                   )
            st.download_button(
                label="Download TekstuStilu.pdf",
                data=pdf_data,
                file_name="TekstuStilu.pdf",
                mime="application/octet-stream",
                )
        except:
            st.write("Za pdf fajl restartujte app za 5 minuta. Osvezavanje aplikacije je u toku")

        st.download_button(
            label="Download Odgovor.txt",
            data=st.session_state["odgovor"],
            file_name="Odgovor.txt",
            )
        st.download_button(
            label="Download Odgovor.docx",
            data=html2docx(content=html, title="Zapisnik").getvalue(),
            file_name="Odgovor.docx",
            mime="docx",
            )

if environ.get("DEPLOYMENT_ENVIRONMENT") == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
elif __name__ == "__main__":
    main()
