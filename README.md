# Note Agent – Week 1 CRUD MVP

An AI-powered note management system. This is the Week 1 deliverable: infrastructure setup, authentication, and basic notes CRUD.

## Quick Start

```bash
docker compose up --build
```

The API will be available at **http://localhost:8000**

Interactive docs at **http://localhost:8000/docs**

## API Endpoints

### Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register a new user |
| POST | `/auth/login` | Login (returns JWT token) |

### Workspaces

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/workspaces` | List your workspaces |
| POST | `/workspaces` | Create a workspace |
| PATCH | `/workspaces/{id}` | Rename a workspace |
| DELETE | `/workspaces/{id}` | Delete a workspace |

### Notes CRUD

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/workspaces/{id}/notes` | List notes (paginated) |
| POST | `/workspaces/{id}/notes` | Create a note |
| GET | `/workspaces/{id}/notes/{note_id}` | Get a single note |
| PATCH | `/workspaces/{id}/notes/{note_id}` | Update a note |
| DELETE | `/workspaces/{id}/notes/{note_id}` | Delete a note |

## Tech Stack

- **FastAPI** – async Python web framework
- **PostgreSQL 16** – database
- **SQLAlchemy 2.0** – async ORM
- **JWT** – authentication
- **Docker Compose** – orchestration

