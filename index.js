import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { GoogleGenerativeAI } from '@google/generative-ai';
import puppeteer from 'puppeteer';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Initialize Gemini AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

// Function to extract content from URL
async function extractContentFromURL(url) {
  try {
    const browser = await puppeteer.launch({ headless: 'new' });
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'networkidle0' });
    
    const content = await page.evaluate(() => {
      const article = document.querySelector('article') || 
                     document.querySelector('.article') || 
                     document.querySelector('.post') ||
                     document.querySelector('main');
      
      return article ? article.innerText : document.body.innerText;
    });
    
    await browser.close();
    return content;
  } catch (error) {
    console.error('Error extracting content from URL:', error);
    throw error;
  }
}

// Function to generate summary
async function generateSummary(text) {
  try {
    const prompt = `Please provide a concise summary of the following text, highlighting the main topics and key points:\n\n${text}`;
    console.log('Sending request to Gemini API...');
    const result = await model.generateContent(prompt);
    console.log('Received response from Gemini API');
    const response = await result.response;
    return response.text();
  } catch (error) {
    console.error('Detailed error in generateSummary:', {
      name: error.name,
      message: error.message,
      stack: error.stack,
      details: error.details || 'No additional details'
    });
    throw error;
  }
}

// Function to split text into chunks
function splitTextIntoChunks(text, chunkSize = 1000) {
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
  const chunks = [];
  let currentChunk = '';

  for (const sentence of sentences) {
    if (currentChunk.length + sentence.length > chunkSize) {
      chunks.push(currentChunk.trim());
      currentChunk = sentence;
    } else {
      currentChunk += ' ' + sentence;
    }
  }

  if (currentChunk) {
    chunks.push(currentChunk.trim());
  }

  return chunks;
}

// Helper to extract JSON from markdown code block
function extractJsonFromMarkdown(text) {
  const match = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (match) {
    return match[1];
  }
  const braceMatch = text.match(/{[\s\S]*}/);
  if (braceMatch) {
    return braceMatch[0];
  }
  return text;
}

// Function to extract topics
async function extractTopics(text) {
  try {
    const chunks = splitTextIntoChunks(text);
    const topics = [];
    
    for (const chunk of chunks) {
      const prompt = `
You are an expert at breaking down information into mind maps.

Given the following text, extract the main topic, and then break it down into 3-7 key subtopics. For each subtopic, if possible, break it down further into sub-subtopics. 

Return the result as a valid JSON object in this format (no explanation, just JSON):

{
  "title": "Main Topic",
  "content": "Short summary of the main topic",
  "children": [
    {
      "title": "Subtopic 1",
      "content": "Summary of subtopic 1",
      "children": [
        {
          "title": "Sub-subtopic 1",
          "content": "Summary of sub-subtopic 1",
          "children": []
        }
      ]
    },
    {
      "title": "Subtopic 2",
      "content": "Summary of subtopic 2",
      "children": []
    }
  ]
}

Text to analyze:
${chunk}

Return only the JSON, no explanation or extra text.
      `;
      const result = await model.generateContent(prompt);
      const response = await result.response;
      const raw = response.text();
      console.log('Gemini raw output:', raw);
      const jsonString = extractJsonFromMarkdown(raw);
      try {
        const topicData = JSON.parse(jsonString);
        topics.push(topicData);
      } catch (e) {
        console.error('Error parsing topic data:', e, '\nRaw:', jsonString);
      }
    }
    
    return mergeTopics(topics);
  } catch (error) {
    console.error('Error extracting topics:', error);
    throw error;
  }
}

// Function to merge topics
function mergeTopics(topics) {
  const merged = {
    id: 'root',
    title: 'Main Topic',
    content: '',
    level: 0,
    children: []
  };
  
  topics.forEach(topic => {
    if (!merged.content) {
      merged.content = topic.content;
    }
    
    topic.children?.forEach(child => {
      const existingChild = merged.children.find(c => 
        c.title.toLowerCase() === child.title.toLowerCase()
      );
      
      if (!existingChild) {
        merged.children.push({
          ...child,
          id: `topic-${merged.children.length + 1}`,
          level: 1,
          children: child.children || []
        });
      }
    });
  });
  
  return merged;
}

// Function to convert hierarchy to mind map format
function hierarchyToMindMap(hierarchy) {
  const nodes = [];
  const edges = [];
  
  const processNode = (node, parentId, x = 0, y = 0) => {
    const nodeData = {
      id: node.id,
      type: 'custom',
      position: { x, y },
      data: {
        label: node.title,
        level: node.level,
        content: node.content,
        isExpanded: true
      }
    };
    
    nodes.push(nodeData);
    
    if (parentId) {
      edges.push({
        id: `${parentId}-${node.id}`,
        source: parentId,
        target: node.id,
        type: 'smoothstep',
        animated: true,
        style: {
          stroke: node.level === 1 ? '#10b981' : '#f97316',
          strokeWidth: node.level === 1 ? 3 : 2,
        }
      });
    }
    
    const childSpacing = 200;
    const startX = x - (node.children.length - 1) * childSpacing / 2;
    
    node.children.forEach((child, index) => {
      const childX = startX + index * childSpacing;
      const childY = y + 150;
      processNode(child, node.id, childX, childY);
    });
  };
  
  processNode(hierarchy);
  return { nodes, edges };
}

// API endpoint for processing content
app.post('/api/process', async (req, res) => {
  try {
    const { content, type } = req.body;
    
    if (!content) {
      return res.status(400).json({ error: 'Content is required' });
    }

    // Check if API key is set
    if (!process.env.GEMINI_API_KEY) {
      console.error('GEMINI_API_KEY is not set in environment variables');
      return res.status(500).json({ 
        error: 'Server configuration error',
        details: 'API key is not configured'
      });
    }
    
    let text = content;
    
    // If URL, extract content first
    if (type === 'url') {
      text = await extractContentFromURL(content);
    }
    
    // Generate summary
    const summary = await generateSummary(text);
    
    // Extract topics
    const topics = await extractTopics(text);
    
    // Update root content with summary
    topics.content = summary;
    
    // Convert to mind map format
    const mindMap = hierarchyToMindMap(topics);
    
    res.json(mindMap);
  } catch (error) {
    console.error('Detailed error in /api/process:', {
      name: error.name,
      message: error.message,
      stack: error.stack,
      details: error.details || 'No additional details'
    });
    
    res.status(500).json({ 
      error: 'Failed to process content',
      details: error.message,
      type: error.name
    });
  }
});

app.get('/', (req, res) => {
  res.send('The backend is working fine babygurl');
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
}); 