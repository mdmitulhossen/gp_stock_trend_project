import streamlit as st
import requests
import re
import streamlit.components.v1 as components


# -----------------------------
# Your OpenRouter API KEY
# -----------------------------
OPENROUTER_API_KEY = "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"  # your actual key here

# -----------------------------
# Helper: Highlight text with emoji
# -----------------------------
def highlight_text_with_emoji(text):
    text = re.sub(r'\bGP\b', '📈 **GP**', text, flags=re.IGNORECASE)
    text = re.sub(r'\bstock\b', '💹 stock', text, flags=re.IGNORECASE)
    text = re.sub(r'\bprofit\b', '💰 profit', text, flags=re.IGNORECASE)
    text = re.sub(r'\bgrowth\b', '🚀 growth', text, flags=re.IGNORECASE)
    return text

# -----------------------------
# Mermaid chart renderer
# -----------------------------
def render_mermaid_chart(mermaid_code):
    html_code = f"""
    <div style="background:#1e1e1e; padding:15px; border-radius:8px;">
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <div class="mermaid">
        {mermaid_code}
        </div>
        <script>
        mermaid.initialize({{startOnLoad:true}});
        </script>
    </div>
    """
    components.html(html_code, height=300, scrolling=True)

# -----------------------------
# Optimized stream with markdown blocks
# -----------------------------
def stream_with_markdown_and_summary_v2(text):
    lines = text.split("\n")
    block = []
    summary = []
    block_type = None

    def render_block(btype, content):
        joined = "\n".join(content)
        if btype in ["table", "heading"]:
            st.markdown(joined)
        else:
            for line in content:
                line_hl = highlight_text_with_emoji(line)
                if re.search(r'\bbuy\b', line, re.IGNORECASE):
                    st.markdown(f"<p style='color:#00ff99;'>{line_hl}</p>", unsafe_allow_html=True)
                elif re.search(r'\bsell\b', line, re.IGNORECASE):
                    st.markdown(f"<p style='color:#ff6666;'>{line_hl}</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='color:#ccc;'>{line_hl}</p>", unsafe_allow_html=True)
                summary.append(line_hl)

    for line in lines:
        line_strip = line.strip()

        # Mermaid block
        if line_strip.startswith("```mermaid"):
            block_type = "mermaid"
            block = []
            continue
        elif block_type == "mermaid" and line_strip == "```":
            render_mermaid_chart("\n".join(block))
            block_type = None
            continue
        elif block_type == "mermaid":
            block.append(line)
            continue

        # Table block
        if line_strip.startswith("|"):
            if block_type != "table":
                if block:
                    render_block(block_type, block)
                block = []
                block_type = "table"
            block.append(line)
        # Headings or bullet or bold
        elif line_strip.startswith("###") or line_strip.startswith("- ") or line_strip.startswith("**"):
            if block_type != "heading":
                if block:
                    render_block(block_type, block)
                block = []
                block_type = "heading"
            block.append(line)
        # Empty line -> flush block
        elif line_strip == "":
            if block:
                render_block(block_type, block)
                block = []
                block_type = None
        else:
            if block_type != "normal":
                if block:
                    render_block(block_type, block)
                block = []
                block_type = "normal"
            block.append(line)

    if block:
        render_block(block_type, block)

    # Summary
    summary_tail = summary[-5:]
    st.markdown(f"""
    <div style="background: rgba(0,255,204,0.08); padding: 15px; margin-top:20px;
                border-left: 4px solid #00ffcc; border-radius: 8px;">
        <h4 style="color:#00ffcc; margin:0;">🔎 Summary:</h4>
        <p style="color:#ccc;">{" ".join(summary_tail)}</p>
    </div>
    """, unsafe_allow_html=True)

def show_analysis_with_ai():
        # -----------------------------
        # Streamlit UI
        # -----------------------------
        st.title("🤖 Analysis with AI (GP Focused)")

        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.05); padding: 30px; 
                        border-radius: 12px; border: 2px solid #00ffcc;
                        box-shadow: 0 0 20px #00ffcc33; text-align:center;">
                <h3 style="color:#00ffcc;">💬 GP-focused AI Chatbot</h3>
                <p style="color:#aaa;">Ask anything about Grameenphone (GP) or Bangladesh stock market. 
                Our AI responds with data-driven insights, markdown tables and mermaid charts.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -----------------------------
        # User input
        # -----------------------------
        user_query = st.text_input(
            "Ask your question here...",
            placeholder="E.g. Should I buy GP stock now?",
            key="chat_input"
        )

        # -----------------------------
        # On user submit
        # -----------------------------
        if user_query:
            with st.spinner("Thinking... 🤔"):
                prompt = (
                    f"You are an expert on Grameenphone (GP) and Bangladesh stock market. "
                    f"Answer ONLY about GP or the Bangladesh stock context. "
                    f"If relevant, use markdown tables or mermaid charts. "
                    f"Question: {user_query}"
                )

                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": "tngtech/deepseek-r1t2-chimera:free",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                }

                try:
                    res = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    res.raise_for_status()
                    ai_text = res.json()['choices'][0]['message']['content']
                except Exception as e:
                    ai_text = f"🚨 Error contacting AI service: {e}"

            st.markdown("""
                <div style="background: rgba(0,255,204,0.1); padding:20px; 
                            border-radius:8px; margin-top:20px;
                            border:1px solid #00ffcc;">
            """, unsafe_allow_html=True)

            stream_with_markdown_and_summary_v2(ai_text)

            st.markdown("</div>", unsafe_allow_html=True)