from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, ForeignKey, DateTime, JSON, Index, select, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session, sessionmaker
from backend.app.domain import models
from backend.app.domain.ports import MetadataStore
from backend.app.config.schema import AppConfig

# --- SQLAlchemy Models ---

class Base(DeclarativeBase):
    pass

class SourceORM(Base):
    __tablename__ = "sources"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    config: Mapped[dict] = mapped_column(JSON, default={})
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_domain(self) -> models.Source:
        return models.Source(
            id=UUID(self.id),
            name=self.name,
            path=self.path,
            config=self.config,
            created_at=self.created_at,
            updated_at=self.updated_at
        )

class DocumentORM(Base):
    __tablename__ = "documents"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_id: Mapped[str] = mapped_column(String, ForeignKey("sources.id"), nullable=False, index=True)
    uri: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mtime: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    doc_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="new")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_domain(self) -> models.Document:
        return models.Document(
            id=UUID(self.id),
            source_id=UUID(self.source_id),
            uri=self.uri,
            title=self.title,
            mime_type=self.mime_type,
            size_bytes=self.size_bytes,
            mtime=self.mtime,
            doc_hash=self.doc_hash,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at
        )

class ChunkORM(Base):
    __tablename__ = "chunks"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(String, nullable=False)
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_domain(self) -> models.Chunk:
        return models.Chunk(
            id=UUID(self.id),
            doc_id=UUID(self.doc_id),
            chunk_index=self.chunk_index,
            text=self.text,
            start_offset=self.start_offset,
            end_offset=self.end_offset,
            chunk_hash=self.chunk_hash,
            created_at=self.created_at,
            updated_at=self.updated_at
        )

class JobORM(Base):
    __tablename__ = "jobs"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default={})
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_domain(self) -> models.Job:
        return models.Job(
            id=UUID(self.id),
            type=models.JobType(self.type),
            status=models.JobStatus(self.status),
            progress=self.progress,
            error=self.error,
            payload=self.payload,
            created_at=self.created_at,
            updated_at=self.updated_at
        )

# --- Implementation ---

class SQLiteMetadataStore(MetadataStore):
    def __init__(self, config: AppConfig):
        self.db_path = config.storage.sqlite_path
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Simple schema initialization (no alembic for now per deliverables)
        Base.metadata.create_all(bind=self.engine)

    def upsert_source(self, source: models.Source) -> models.Source:
        with self.SessionLocal() as session:
            orm = session.get(SourceORM, str(source.id))
            if not orm:
                orm = SourceORM(id=str(source.id))
                session.add(orm)
            
            orm.name = source.name
            orm.path = source.path
            orm.config = source.config
            orm.updated_at = datetime.utcnow()
            
            session.commit()
            session.refresh(orm)
            return orm.to_domain()

    def get_source(self, source_id: UUID) -> Optional[models.Source]:
        with self.SessionLocal() as session:
            orm = session.get(SourceORM, str(source_id))
            return orm.to_domain() if orm else None

    def list_sources(self) -> List[models.Source]:
        with self.SessionLocal() as session:
            orms = session.execute(select(SourceORM)).scalars().all()
            return [orm.to_domain() for orm in orms]

    def upsert_document(self, doc: models.Document) -> models.Document:
        with self.SessionLocal() as session:
            orm = session.get(DocumentORM, str(doc.id))
            if not orm:
                orm = DocumentORM(id=str(doc.id))
                session.add(orm)
            
            orm.source_id = str(doc.source_id)
            orm.uri = doc.uri
            orm.title = doc.title
            orm.mime_type = doc.mime_type
            orm.size_bytes = doc.size_bytes
            orm.mtime = doc.mtime
            orm.doc_hash = doc.doc_hash
            orm.status = doc.status
            orm.updated_at = datetime.utcnow()
            
            session.commit()
            session.refresh(orm)
            return orm.to_domain()

    def get_document(self, doc_id: UUID) -> Optional[models.Document]:
        with self.SessionLocal() as session:
            orm = session.get(DocumentORM, str(doc_id))
            return orm.to_domain() if orm else None

    def list_documents_by_source(self, source_id: UUID) -> List[models.Document]:
        with self.SessionLocal() as session:
            stmt = select(DocumentORM).where(DocumentORM.source_id == str(source_id))
            orms = session.execute(stmt).scalars().all()
            return [orm.to_domain() for orm in orms]
    
    def mark_document_deleted(self, doc_id: UUID) -> None:
        # We can actually delete it or mark status='deleted'
        # Requirements imply "mark_deleted", usually status update
        with self.SessionLocal() as session:
            orm = session.get(DocumentORM, str(doc_id))
            if orm:
                orm.status = "deleted"
                orm.updated_at = datetime.utcnow()
                session.commit()

    def upsert_chunk(self, chunk: models.Chunk) -> models.Chunk:
        with self.SessionLocal() as session:
            orm = session.get(ChunkORM, str(chunk.id))
            if not orm:
                orm = ChunkORM(id=str(chunk.id))
                session.add(orm)
            
            orm.doc_id = str(chunk.doc_id)
            orm.chunk_index = chunk.chunk_index
            orm.text = chunk.text
            orm.start_offset = chunk.start_offset
            orm.end_offset = chunk.end_offset
            orm.chunk_hash = chunk.chunk_hash
            orm.updated_at = datetime.utcnow()
            
            session.commit()
            session.refresh(orm)
            return orm.to_domain()

    def list_chunks(self, doc_id: UUID) -> List[models.Chunk]:
        with self.SessionLocal() as session:
            stmt = select(ChunkORM).where(ChunkORM.doc_id == str(doc_id)).order_by(ChunkORM.chunk_index)
            orms = session.execute(stmt).scalars().all()
            return [orm.to_domain() for orm in orms]

    def get_chunk(self, chunk_id: UUID) -> Optional[models.Chunk]:
        with self.SessionLocal() as session:
            orm = session.get(ChunkORM, str(chunk_id))
            return orm.to_domain() if orm else None

    def upsert_job(self, job: models.Job) -> models.Job:
        with self.SessionLocal() as session:
            orm = session.get(JobORM, str(job.id))
            if not orm:
                orm = JobORM(id=str(job.id))
                session.add(orm)
            
            orm.type = job.type.value
            orm.status = job.status.value
            orm.progress = job.progress
            orm.error = job.error
            orm.payload = job.payload
            orm.updated_at = datetime.utcnow()
            
            session.commit()
            session.refresh(orm)
            return orm.to_domain()

    def get_job(self, job_id: UUID) -> Optional[models.Job]:
        with self.SessionLocal() as session:
            orm = session.get(JobORM, str(job_id))
            return orm.to_domain() if orm else None

    def list_jobs(self) -> List[models.Job]:
        with self.SessionLocal() as session:
            stmt = select(JobORM).order_by(JobORM.created_at.desc())
            orms = session.execute(stmt).scalars().all()
            return [orm.to_domain() for orm in orms]

    def get_pending_jobs(self, limit: int = 10) -> List[models.Job]:
        with self.SessionLocal() as session:
            stmt = select(JobORM).where(JobORM.status == models.JobStatus.PENDING.value).order_by(JobORM.created_at.asc()).limit(limit)
            orms = session.execute(stmt).scalars().all()
            return [orm.to_domain() for orm in orms]
