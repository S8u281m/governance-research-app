# 🔬 Governance Research Assistant - Setup Guide

## 🚀 Quick Start

### Local Development

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the App**
```bash
streamlit run governance_research_app.py
```

3. **Open in Browser**
- App will automatically open at `http://localhost:8501`

### 🌐 Free Web Deployment (Streamlit Community Cloud)

1. **Create GitHub Repository**
   - Upload `governance_research_app.py` and `requirements.txt` to GitHub
   - Make repository public

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and main file (`governance_research_app.py`)
   - Click "Deploy!"

3. **Access Your App**
   - Get a permanent URL like `https://yourapp.streamlit.app`
   - Share with collaborators

## 📋 Features Overview

### 🏠 Dashboard
- Project overview and statistics
- Recent activity tracking
- Quick metrics display

### 📄 Upload Documents
- **Training Data**: Upload Atlas.ti .qdpx files
- **Transcripts**: Upload interview .docx files
- Automatic text extraction and validation

### 🤖 AI Analysis
- Integration with your existing analysis pipeline
- Real-time progress tracking
- Claude API integration for governance coding

### ✏️ Review Codes
- Interactive code editing interface
- Filter and search functionality
- Approve/reject AI-generated codes
- Add manual codes

### 📊 Visualizations
- **Code Frequency**: Bar charts and distributions
- **Relationships**: Co-occurrence matrices and heatmaps
- **Document Comparison**: Cross-interview analysis
- **Interactive Plots**: Zoom, filter, and explore

### 📋 Export Results
- CSV and Excel download options
- Multi-sheet Excel with summaries
- Publication-ready formatting

## 🔧 Configuration

### API Keys
- Enter Claude API key directly in the app interface
- Keys are not stored permanently (enter each session)

### Data Storage
- Session state storage (temporary)
- For persistent storage, can be extended with SQLite database

## 💡 Usage Tips

1. **Start with Training Data**: Upload your Atlas.ti file first
2. **Upload Transcripts**: Add interview documents for analysis
3. **Review AI Results**: Always validate AI-generated codes
4. **Use Visualizations**: Explore patterns across documents
5. **Export Regularly**: Download results for backup

## 🛠️ Customization Options

### Adding New Visualizations
- Modify the `visualization_page()` function
- Use Plotly for interactive charts
- Add NetworkX for network analysis

### Extending Code Categories
- Update the code selection dropdown in review interface
- Modify the analysis prompts for new domains

### Database Integration
- Replace session state with SQLite for persistence
- Add user authentication for multi-researcher support

## 📈 Next Steps

1. **Test the Basic Workflow**: Upload → Analyze → Review → Export
2. **Customize Visualizations**: Add domain-specific charts
3. **Deploy Online**: Share with research team
4. **Iterate and Improve**: Add features based on usage

## 🆘 Troubleshooting

### Common Issues

**"API Error"**
- Check Claude API key is valid
- Verify internet connection
- Check API rate limits

**"No chunks created"**
- Verify transcript format (speaker names + timestamps)
- Check document encoding

**"Upload failed"**
- Ensure file formats (.qdpx for Atlas.ti, .docx for transcripts)
- Check file size limits

### Getting Help
- Check Streamlit documentation: https://docs.streamlit.io/
- Review error messages in the browser console
- Test with sample data first

## 🎯 Research Workflow

1. **Preparation**
   - Upload Atlas.ti training data
   - Verify code extraction

2. **Analysis**
   - Upload new interview transcripts
   - Run AI analysis with Claude
   - Review confidence scores

3. **Validation**
   - Review AI-generated codes
   - Edit/approve/reject as needed
   - Add manual codes if necessary

4. **Exploration**
   - Use visualizations to explore patterns
   - Compare across documents
   - Identify relationships

5. **Export**
   - Download results for further analysis
   - Generate publication-ready outputs
   - Share with research team

🚀 **Ready to transform your qualitative research workflow!**
