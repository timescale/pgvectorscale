#!/bin/bash
set -e

# Default values
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_USER=${DB_USER:-$USER}
DB_NAME=${DB_NAME:-postgres}
PYTEST_ARGS=${PYTEST_ARGS:-"-v"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔧 Setting up Python test environment...${NC}"

# Check if PostgreSQL is running
if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; then
    echo -e "${RED}❌ PostgreSQL is not running or not accessible${NC}"
    echo "Please ensure PostgreSQL is running and accessible at:"
    echo "  Host: $DB_HOST"
    echo "  Port: $DB_PORT" 
    echo "  User: $DB_USER"
    echo "  Database: $DB_NAME"
    echo ""
    echo "For PGRX development, try:"
    echo "  cd pgvectorscale && cargo pgrx start pg17"
    echo "  DB_PORT=28817 $0"
    exit 1
fi

# Check if we're in the correct directory
if [ ! -f "tests/requirements.txt" ]; then
    echo -e "${RED}❌ tests/requirements.txt not found${NC}"
    echo "Please run this script from the pgvectorscale root directory"
    exit 1
fi

echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"

# Check if we have a .venv directory (created by make target)
if [ -d ".venv" ]; then
    echo "Using project virtual environment: .venv"
    .venv/bin/pip install -r tests/requirements.txt >/dev/null 2>&1
elif [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Using active virtual environment: $VIRTUAL_ENV"
    pip install -r tests/requirements.txt
elif command -v pip3 >/dev/null 2>&1; then
    # Try pip3 with --user flag first
    if pip3 install --user -r tests/requirements.txt >/dev/null 2>&1; then
        echo "Installed dependencies with pip3 --user"
    else
        echo -e "${RED}❌ Could not install Python dependencies${NC}"
        echo "Please create a virtual environment and install dependencies:"
        echo "  python3 -m venv .venv"
        echo "  .venv/bin/pip install -r tests/requirements.txt"
        echo "Or run: make test-python-setup"
        exit 1
    fi
else
    echo -e "${RED}❌ pip3 not found${NC}"
    echo "Please install Python 3 and pip"
    exit 1
fi

# Ensure extensions are installed
echo -e "${YELLOW}🔌 Checking extensions...${NC}"
DATABASE_URL="postgresql+asyncpg://$DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"

if ! python3 -c "
import asyncio
import asyncpg
import sys

async def check_extensions():
    try:
        conn = await asyncpg.connect('postgresql://$DB_USER@$DB_HOST:$DB_PORT/$DB_NAME')
        try:
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            print('✅ pgvector extension ready')
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vectorscale')
            print('✅ pgvectorscale extension ready')
        except Exception as e:
            print(f'❌ Extension setup failed: {e}')
            print('Please ensure pgvector and pgvectorscale are installed and available')
            sys.exit(1)
        finally:
            await conn.close()
    except Exception as e:
        print(f'❌ Database connection failed: {e}')
        sys.exit(1)

asyncio.run(check_extensions())
"; then
    exit 1
fi

# Run tests
echo -e "${GREEN}🧪 Running Python tests...${NC}"
export DATABASE_URL="$DATABASE_URL"

# Use pytest from virtual environment if available, otherwise fallback
if [ -d ".venv" ]; then
    .venv/bin/python -m pytest tests/ "$PYTEST_ARGS"
elif command -v pytest >/dev/null 2>&1; then
    pytest tests/ "$PYTEST_ARGS"
else
    python3 -m pytest tests/ "$PYTEST_ARGS"
fi

echo -e "${GREEN}✅ Python tests completed!${NC}"