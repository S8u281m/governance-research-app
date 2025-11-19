import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from plotly.subplots import make_subplots
import zipfile
import xml.etree.ElementTree as ET
import docx
from docx import Document
import anthropic
import re
import time
import sqlite3
import os
from typing import List, Dict
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Governance Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = pd.DataFrame()
if 'training_data' not in st.session_state:
    st.session_state.training_data = pd.DataFrame()
if 'projects' not in st.session_state:
    st.session_state.projects = {}

# ===== CORE ANALYSIS FUNCTIONS (From your Colab) =====

def extract_atlas_quotations(qdpx_path):
    """
    Extract coded quotations from Atlas.ti .qdpx file
    """
    training_data = []

    with zipfile.ZipFile(qdpx_path, 'r') as zip_ref:
        project_content = zip_ref.read('project.qde')
        root = ET.fromstring(project_content)

        ns = {'qda': 'urn:QDA-XML:project:1.0'}

        # Extract codes
        codes = {}
        codebook = root.find('qda:CodeBook', ns)
        if codebook is not None:
            all_codes = codebook.findall('.//qda:Code', ns)
            for code in all_codes:
                code_guid = code.get('guid')
                code_name = code.get('name', 'Unknown')
                codes[code_guid] = code_name

        # Extract sources
        sources = {}
        sources_elem = root.find('qda:Sources', ns)
        if sources_elem is not None:
            text_sources = sources_elem.findall('qda:TextSource', ns)
            for source in text_sources:
                source_guid = source.get('guid')
                source_name = source.get('name', 'Unknown')
                sources[source_guid] = source_name

        # Extract quotations
        for source_guid, source_name in sources.items():
            text_source = sources_elem.find(f".//qda:TextSource[@guid='{source_guid}']", ns)
            if text_source is not None:
                selections = text_source.findall('qda:PlainTextSelection', ns)

                for selection in selections:
                    try:
                        quotation_text = selection.get('name', '').strip()

                        if quotation_text:
                            codings = selection.findall('qda:Coding', ns)

                            for coding in codings:
                                code_ref = coding.find('qda:CodeRef', ns)
                                if code_ref is not None:
                                    code_guid = code_ref.get('targetGUID')
                                    code_name = codes.get(code_guid, 'Unknown Code')

                                    training_data.append({
                                        'document': source_name,
                                        'quotation': quotation_text,
                                        'code': code_name,
                                        'confidence': 1.0,
                                        'source': 'atlas_ti'
                                    })

                    except Exception as e:
                        continue

    return pd.DataFrame(training_data)

def chunk_interview_transcript(text: str, max_qa_pairs: int = 2) -> List[Dict]:
    """
    Chunk interview transcript into Q&A pairs
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    chunks = []
    current_qa_pairs_for_chunk = []

    speaker_pattern = re.compile(r'^(.*?)\s+(\d+:\d+)$')

    current_speaker_name = None
    current_speaker_dialogue = []
    last_question = ""

    for i, line in enumerate(lines):
        if i < 2 and any(skip_word in line.lower() for skip_word in ['transcript', 'started transcription', 'pm', 'am']):
            continue

        match = speaker_pattern.match(line)
        if match:
            speaker_name_raw = match.group(1).strip()
            
            if current_speaker_name is not None and current_speaker_dialogue:
                dialogue_text = " ".join(current_speaker_dialogue).strip()

                if 'shubham' in current_speaker_name.lower() or 'sharma' in current_speaker_name.lower():
                    last_question = dialogue_text
                else:
                    if last_question:
                        current_qa_pairs_for_chunk.append({
                            'context_question': last_question,
                            'respondent_answer': dialogue_text
                        })
                        last_question = ""

                        if len(current_qa_pairs_for_chunk) >= max_qa_pairs:
                            chunk_text_content = ""
                            for qa_pair in current_qa_pairs_for_chunk:
                                chunk_text_content += f"CONTEXT: {qa_pair['context_question']}\n"
                                chunk_text_content += f"RESPONSE: {qa_pair['respondent_answer']}\n\n"
                            
                            chunks.append({
                                'respondent_segments': current_qa_pairs_for_chunk.copy(),
                                'text': chunk_text_content.strip(),
                                'word_count': len(chunk_text_content.split())
                            })
                            current_qa_pairs_for_chunk = []

            current_speaker_name = speaker_name_raw
            current_speaker_dialogue = []
        else:
            current_speaker_dialogue.append(line)

    # Handle final segments
    if current_speaker_name is not None and current_speaker_dialogue:
        dialogue_text = " ".join(current_speaker_dialogue).strip()
        if 'shubham' in current_speaker_name.lower() or 'sharma' in current_speaker_name.lower():
            last_question = dialogue_text
        else:
            if last_question:
                current_qa_pairs_for_chunk.append({
                    'context_question': last_question,
                    'respondent_answer': dialogue_text
                })

    if current_qa_pairs_for_chunk:
        chunk_text_content = ""
        for qa_pair in current_qa_pairs_for_chunk:
            chunk_text_content += f"CONTEXT: {qa_pair['context_question']}\n"
            chunk_text_content += f"RESPONSE: {qa_pair['respondent_answer']}\n\n"

        chunks.append({
            'respondent_segments': current_qa_pairs_for_chunk.copy(),
            'text': chunk_text_content.strip(),
            'word_count': len(chunk_text_content.split())
        })

    return chunks

def analyze_governance_patterns_fixed(text_chunk: str, training_examples: pd.DataFrame) -> str:
    """
    Analyze governance patterns without contamination
    """
    if training_examples.empty:
        return "Error: No training data available"
    
    top_codes = training_examples['code'].value_counts().head(12).index.tolist()
    
    reference_context = "REFERENCE CODE PATTERNS (for pattern recognition only - DO NOT quote from these):\n\n"
    
    for code in top_codes:
        reference_context += f"• {code}\n"
    
    reference_context += "\n" + "="*60 + "\n"

    prompt = f"""{reference_context}

CRITICAL INSTRUCTIONS:
1. ONLY extract quotes from the NEW INTERVIEW TEXT below
2. DO NOT quote from the reference patterns above
3. Use the reference patterns to identify similar themes in the NEW TEXT
4. Extract quotes EXCLUSIVELY from the interview segment provided

NEW INTERVIEW TO ANALYZE:
{text_chunk}

For each relevant theme you identify in the NEW INTERVIEW:
1. Extract exact quote (20+ words minimum) FROM THE INTERVIEW ONLY
2. Assign most appropriate code from the reference patterns
3. Explain reasoning
4. Rate confidence (1-5)

Use this format:
**CODE: [code name]**
Quote: "[exact text from NEW INTERVIEW only]"
Reasoning: [why this code fits]
Confidence: [1-5]

REMEMBER: Extract quotes ONLY from the interview text above, never from reference patterns."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
        
    except Exception as e:
        return f"API Error: {str(e)}"

def parse_claude_analysis_to_codes(analysis_text: str, chunk_id: int, interview: str) -> List[Dict]:
    """
    Parse Claude's analysis into structured codes
    """
    results = []
    
    pattern = r'\*\*CODE:\s*([^*]+?)\*\*\s*Quote:\s*"([^"]+?)"\s*Reasoning:\s*([^*]+?)(?:Confidence:\s*(\d))?'
    
    matches = re.findall(pattern, analysis_text, re.DOTALL)
    
    for match in matches:
        code = match[0].strip()
        quote = match[1].strip()
        reasoning = match[2].strip()
        confidence = int(match[3]) if match[3] else 3
        
        results.append({
            'code': code,
            'quote': quote,
            'reasoning': reasoning,
            'confidence': confidence,
            'chunk_id': chunk_id,
            'interview': interview,
            'quote_length': len(quote.split()),
            'status': 'AI Generated'  # For review interface
        })
    
    return results

# ===== STREAMLIT APP INTERFACE =====

def sidebar_navigation():
    """Create sidebar navigation"""
    st.sidebar.title("🔬 Research Assistant")
    
    pages = {
        "🏠 Dashboard": "dashboard",
        "📄 Upload Documents": "upload",
        "🤖 AI Analysis": "analysis",
        "✏️ Review Codes": "review",
        "📊 Visualizations": "viz",
        "📋 Export Results": "export"
    }
    
    selected = st.sidebar.radio("Navigate", list(pages.keys()))
    return pages[selected]

def dashboard_page():
    """Project dashboard and overview"""
    st.title("🏠 Governance Research Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Examples", len(st.session_state.training_data))
    
    with col2:
        st.metric("Analyzed Documents", 
                 len(st.session_state.analysis_results['interview'].unique()) if not st.session_state.analysis_results.empty else 0)
    
    with col3:
        st.metric("Total Coded Quotes", len(st.session_state.analysis_results))
    
    with col4:
        st.metric("Unique Codes", 
                 len(st.session_state.analysis_results['code'].unique()) if not st.session_state.analysis_results.empty else 0)
    
    st.markdown("---")
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    if not st.session_state.analysis_results.empty:
        recent_codes = st.session_state.analysis_results.tail(10)[['interview', 'code', 'confidence']]
        st.dataframe(recent_codes)
    else:
        st.info("No analysis results yet. Upload documents and run AI analysis to get started!")

def upload_page():
    """Document upload and training data management"""
    st.title("📄 Document Management")
    
    tab1, tab2 = st.tabs(["📊 Training Data", "📄 Interview Transcripts"])
    
    with tab1:
        st.subheader("Upload Atlas.ti Training Data")
        atlas_file = st.file_uploader("Choose .qdpx file", type=['qdpx'])
        
        if atlas_file is not None:
            with st.spinner("Extracting training data..."):
                # Save uploaded file temporarily
                with open("temp_atlas.qdpx", "wb") as f:
                    f.write(atlas_file.getbuffer())
                
                # Extract training data
                training_df = extract_atlas_quotations("temp_atlas.qdpx")
                st.session_state.training_data = training_df
                
                # Clean up
                os.remove("temp_atlas.qdpx")
                
                st.success(f"✅ Extracted {len(training_df)} coded quotations!")
                st.write(f"📊 Documents: {training_df['document'].nunique()}")
                st.write(f"🏷️ Unique codes: {training_df['code'].nunique()}")
                
                # Show code frequency
                st.subheader("Code Frequency")
                code_freq = training_df['code'].value_counts().head(10)
                fig = px.bar(x=code_freq.values, y=code_freq.index, orientation='h', 
                           title="Top 10 Most Frequent Codes")
                st.plotly_chart(fig)
    
    with tab2:
        st.subheader("Upload Interview Transcripts")
        interview_file = st.file_uploader("Choose .docx file", type=['docx'])
        
        if interview_file is not None:
            # Extract text from Word document
            doc = Document(interview_file)
            all_text = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    all_text.append(text)
            
            transcript_text = '\n'.join(all_text)
            
            st.success(f"✅ Loaded transcript: {len(transcript_text)} characters")
            st.session_state.current_transcript = {
                'filename': interview_file.name,
                'text': transcript_text
            }
            
            # Preview
            st.subheader("Preview")
            st.text_area("Transcript preview", transcript_text[:1000] + "...", height=200)

def analysis_page():
    """AI analysis interface"""
    st.title("🤖 AI Governance Analysis")
    
    if st.session_state.training_data.empty:
        st.error("Please upload Atlas.ti training data first!")
        return
    
    if 'current_transcript' not in st.session_state:
        st.error("Please upload an interview transcript first!")
        return
    
    # API key input
    api_key = st.text_input("Enter Claude API Key", type="password")
    
    if st.button("🚀 Start Analysis") and api_key:
        transcript = st.session_state.current_transcript
        
        with st.spinner("Analyzing transcript..."):
            # Progress bar
            progress = st.progress(0)
            
            # Chunk the transcript
            chunks = chunk_interview_transcript(transcript['text'], max_qa_pairs=2)
            st.write(f"📊 Created {len(chunks)} analysis chunks")
            
            # Analyze each chunk
            all_coded_quotes = []
            
            for i, chunk in enumerate(chunks):
                progress.progress((i + 1) / len(chunks))
                st.write(f"🤖 Analyzing chunk {i+1}/{len(chunks)}...")
                
                try:
                    analysis = analyze_governance_patterns_fixed(
                        chunk['text'], 
                        st.session_state.training_data, 
                        api_key
                    )
                    
                    coded_quotes = parse_claude_analysis_to_codes(
                        analysis, i+1, transcript['filename']
                    )
                    all_coded_quotes.extend(coded_quotes)
                    
                except Exception as e:
                    st.error(f"Error in chunk {i+1}: {str(e)}")
                
                time.sleep(1)  # Rate limiting
            
            # Store results
            if all_coded_quotes:
                new_results = pd.DataFrame(all_coded_quotes)
                st.session_state.analysis_results = pd.concat([
                    st.session_state.analysis_results, 
                    new_results
                ], ignore_index=True)
                
                st.success(f"✅ Analysis complete! Found {len(all_coded_quotes)} coded quotes")
                
                # Preview results
                st.subheader("Results Preview")
                st.dataframe(new_results[['code', 'quote', 'confidence']].head())
            else:
                st.warning("No coded quotes found. Check your transcript format.")

def review_page():
    """Interactive code review interface"""
    st.title("✏️ Review & Edit Codes")
    
    if st.session_state.analysis_results.empty:
        st.info("No analysis results to review. Run AI analysis first!")
        return
    
    df = st.session_state.analysis_results.copy()
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_interview = st.selectbox("Filter by Interview", 
                                        ['All'] + list(df['interview'].unique()))
    
    with col2:
        selected_code = st.selectbox("Filter by Code", 
                                   ['All'] + list(df['code'].unique()))
    
    with col3:
        min_confidence = st.slider("Minimum Confidence", 1, 5, 1)
    
    # Apply filters
    if selected_interview != 'All':
        df = df[df['interview'] == selected_interview]
    if selected_code != 'All':
        df = df[df['code'] == selected_code]
    df = df[df['confidence'] >= min_confidence]
    
    st.write(f"Showing {len(df)} results")
    
    # Editable data
    edited_df = st.data_editor(
        df[['code', 'quote', 'reasoning', 'confidence', 'status']],
        column_config={
            'code': st.column_config.SelectboxColumn(
                'Code',
                options=list(st.session_state.training_data['code'].unique()),
                required=True
            ),
            'quote': st.column_config.TextColumn('Quote', max_chars=500),
            'reasoning': st.column_config.TextColumn('Reasoning', max_chars=200),
            'confidence': st.column_config.SliderColumn('Confidence', min_value=1, max_value=5),
            'status': st.column_config.SelectboxColumn(
                'Status',
                options=['AI Generated', 'Reviewed', 'Approved', 'Rejected'],
                required=True
            )
        },
        num_rows="dynamic"
    )
    
    if st.button("💾 Save Changes"):
        # Update the main dataframe with edited values
        # This is a simplified version - you'd want more robust updating
        st.session_state.analysis_results.update(edited_df)
        st.success("Changes saved!")

def visualization_page():
    """Visualizations and cross-document analysis"""
    st.title("📊 Visualizations & Analysis")
    
    if st.session_state.analysis_results.empty:
        st.info("No data to visualize. Run analysis first!")
        return
    
    df = st.session_state.analysis_results
    
    tab1, tab2, tab3 = st.tabs(["📈 Code Frequency", "🌐 Relationships", "📋 Document Comparison"])
    
    with tab1:
        st.subheader("Code Frequency Analysis")
        
        # Code frequency bar chart
        code_counts = df['code'].value_counts()
        fig = px.bar(x=code_counts.values, y=code_counts.index, orientation='h',
                    title="Code Frequency Across All Documents")
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence distribution
        fig2 = px.histogram(df, x='confidence', title="Confidence Score Distribution")
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Code Relationships")
        
        # Code co-occurrence matrix (simplified)
        if len(df) > 1:
            interviews = df['interview'].unique()
            codes = df['code'].unique()
            
            # Create co-occurrence matrix
            cooccur = pd.DataFrame(index=codes, columns=codes, dtype=int).fillna(0)
            
            for interview in interviews:
                int_codes = df[df['interview'] == interview]['code'].unique()
                for i, code1 in enumerate(int_codes):
                    for code2 in int_codes[i+1:]:
                        cooccur.loc[code1, code2] += 1
                        cooccur.loc[code2, code1] += 1
            
            # Heatmap
            fig = px.imshow(cooccur.values, 
                          x=cooccur.columns, 
                          y=cooccur.index,
                          title="Code Co-occurrence Matrix")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Document Comparison")
        
        # Code distribution by document
        doc_code_counts = df.groupby(['interview', 'code']).size().reset_index(name='count')
        
        fig = px.bar(doc_code_counts, x='interview', y='count', color='code',
                    title="Code Distribution by Document")
        st.plotly_chart(fig, use_container_width=True)

def export_page():
    """Export and reporting"""
    st.title("📋 Export Results")
    
    if st.session_state.analysis_results.empty:
        st.info("No results to export.")
        return
    
    df = st.session_state.analysis_results
    
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Download CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV file",
                data=csv,
                file_name="governance_analysis_results.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Download Excel"):
            # Create Excel with multiple sheets
            from io import BytesIO
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='All_Codes', index=False)
                
                # Summary sheet
                summary = df.groupby('code').agg({
                    'quote': 'count',
                    'confidence': 'mean'
                }).round(2)
                summary.to_excel(writer, sheet_name='Code_Summary')
            
            output.seek(0)
            st.download_button(
                label="Download Excel file",
                data=output.getvalue(),
                file_name="governance_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Summary report
    st.subheader("Analysis Summary")
    st.write(f"**Total coded quotes:** {len(df)}")
    st.write(f"**Unique codes:** {df['code'].nunique()}")
    st.write(f"**Documents analyzed:** {df['interview'].nunique()}")
    st.write(f"**Average confidence:** {df['confidence'].mean():.1f}")

# ===== MAIN APP =====

def main():
    """Main app function"""
    
    # Sidebar navigation
    page = sidebar_navigation()
    
    # Route to pages
    if page == "dashboard":
        dashboard_page()
    elif page == "upload":
        upload_page()
    elif page == "analysis":
        analysis_page()
    elif page == "review":
        review_page()
    elif page == "viz":
        visualization_page()
    elif page == "export":
        export_page()

if __name__ == "__main__":
    main()
