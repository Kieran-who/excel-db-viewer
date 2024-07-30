import logging
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import numpy as np
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Adjust logging level for the multipart module
logging.getLogger('multipart').setLevel(logging.WARNING)

app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database setup
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def sanitize_column_name(column_name):    
    sanitized = column_name.strip()    
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', sanitized)    
    if not re.match(r'^[a-zA-Z_]', sanitized):
        sanitized = '_' + sanitized    
    return sanitized.lower()

@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    logger.debug(f"Received file: {file.filename}")
    try:
        df = pd.read_excel(file.file)
        df = df.replace({np.nan: None})  # Replace NaN with None
    except Exception as e:
        logger.error(f"Error reading Excel file: {str(e)}")
        return HTMLResponse(content=f"Error reading Excel file: {str(e)}", status_code=422)

    logger.debug(f"Excel file read. Shape: {df.shape}")
    logger.debug(f"First few rows of the dataframe:\n{df.head().to_string()}")

    # Filter out 'Unnamed' columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    logger.debug(f"Filtered dataframe shape: {df.shape}")

    # Sanitize column headers
    df.columns = [sanitize_column_name(col) for col in df.columns]
    logger.debug(f"Sanitized column names: {df.columns.tolist()}")

    # Create table
    table_name = file.filename.split('.')[0].replace(" ", "_").lower()
    columns = [Column("id", Integer, primary_key=True)]
    for col in df.columns:
        if col != "category":
            columns.append(Column(col, String))

    table = Table(table_name, metadata, *columns)
    metadata.create_all(engine)
    logger.debug(f"Table '{table_name}' created or already exists")

    # Insert data
    inserted_rows = 0
    with SessionLocal() as session:
        try:
            for index, row in df.iterrows():
                # Skip the header row by checking if the row contains column names
                if index == 0 or all(str(row[col]) == col for col in df.columns):
                    logger.debug(f"Skipping header row: {row.to_dict()}")
                    continue
                data = {col: str(row[col]) if row[col] is not None else None for col in df.columns if col != "category"}
                if "category" in df.columns:
                    categories = str(row["category"]).split(",") if row["category"] is not None else []
                    for category in categories:
                        data[sanitize_column_name("category")] = category.strip()
                        session.execute(table.insert().values(**data))
                        inserted_rows += 1
                else:
                    session.execute(table.insert().values(**data))
                    inserted_rows += 1

                if index % 100 == 0:
                    logger.debug(f"Inserted {inserted_rows} rows so far...")
                    session.commit()

            session.commit()  # Final commit for any remaining rows
            logger.debug(f"Total rows inserted into '{table_name}' table: {inserted_rows}")

            # Verify data insertion
            result = session.execute(text(f'SELECT * FROM {table_name} LIMIT 5'))
            sample_data = result.fetchall()
            logger.debug(f"Sample data from database:\n{sample_data}")

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error inserting data: {str(e)}")
            return HTMLResponse(content=f"Error inserting data: {str(e)}", status_code=500)
    
    # Render the updated table content
    columns = [col.name for col in table.columns if col.name != "id"]    
    return templates.TemplateResponse("table.html", {"request": request, "columns": columns, "selected_table": table_name, "success": True})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, table: str | None = None, success: bool = False):
    db = next(get_db())
    inspector = inspect(engine)
    table_name = table

    columns = []
    data = []

    try:
        if table_name and table_name in inspector.get_table_names():
            columns = [column["name"] for column in inspector.get_columns(table_name) if not column["name"].startswith('"Unnamed')]
            logger.debug(f"Columns in '{table_name}' table: {columns}")

            quoted_columns = [sanitize_column_name(col) for col in columns]
            query = text(f'SELECT {", ".join(quoted_columns)} FROM {table_name}')
            result = db.execute(query)
            data = [dict(zip(columns, row)) for row in result.fetchall()]
            logger.debug(f"Retrieved {len(data)} rows from '{table_name}' table")

            if len(data) == 0:
                logger.warning(f"No data retrieved from '{table_name}' table")
            else:
                logger.debug(f"First row of data: {data[0]}")
        else:
            logger.warning(f"Table '{table_name}' does not exist")
    except SQLAlchemyError as e:        
        logger.error(f"Database error: {str(e)}")

    # Fetch all table names for dropdown
    table_names = inspector.get_table_names()    

    context = {
        "request": request,
        "columns": columns,        
        "table_names": table_names,
        "selected_table": table_name
    }
    return templates.TemplateResponse("index.html", context)

@app.get("/table-data", response_class=JSONResponse)
async def get_table_data(request: Request, table: str):
    db = next(get_db())
    inspector = inspect(engine)
    columns = []
    data = []
    error_message = None

    try:
        if table and table in inspector.get_table_names():
            columns = [column["name"] for column in inspector.get_columns(table) if not column["name"].startswith('"Unnamed')]
            quoted_columns = [sanitize_column_name(col) for col in columns]
            query = text(f'SELECT {", ".join(quoted_columns)} FROM {table}')
            result = db.execute(query)
            data = [dict(zip(columns, row)) for row in result.fetchall()]
    except SQLAlchemyError as e:
        error_message = f"An error occurred: {str(e)}"
        return JSONResponse(status_code=500, content={"error": error_message})

    return JSONResponse(content={"columns": columns, "data": data})

@app.get("/table", response_class=HTMLResponse)
async def read_root(request: Request, table: str | None = None, success: bool = False):        
    inspector = inspect(engine)
    table_name = table

    columns = []        

    try:
        if table_name and table_name in inspector.get_table_names():
            columns = [column["name"] for column in inspector.get_columns(table_name) if not column["name"].startswith('"Unnamed')]
            logger.debug(f"Columns in '{table_name}' table: {columns}")           
        else:
            logger.warning(f"Table '{table_name}' does not exist")
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")

    context = {
        "request": request,
        "columns": columns,
    }
    return templates.TemplateResponse("table.html", context)

@app.get("/columns/")
async def get_columns(table: str, db: Session = Depends(get_db)):
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        raise HTTPException(status_code=404, detail="Table not found")

    columns = [column["name"] for column in inspector.get_columns(table) if not column["name"].startswith('"Unnamed')]
    return columns

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
