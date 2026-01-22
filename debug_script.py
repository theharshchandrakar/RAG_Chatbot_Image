try:
    print("Testing imports...")
    import langchain
    print(f"✅ LangChain version: {langchain.__version__}")
    
    from langchain.chains import create_history_aware_retriever
    print("✅ Successfully imported: create_history_aware_retriever")
    
    from langchain.chains import create_retrieval_chain
    print("✅ Successfully imported: create_retrieval_chain")
    
    print("\n🎉 Environment is fixed! You can run chat_console.py now.")
    
except ImportError as e:
    print(f"\n❌ Import Failed: {e}")
    print("Please verify you are running this in the correct virtual environment.")