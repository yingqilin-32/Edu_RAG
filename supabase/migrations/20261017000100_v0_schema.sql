-- V0: six domain tables plus a migration ledger. Apply to your Edu RAG project.
-- This file is transactional when run by `python -m edu_rag.db migrate`.
-- It never drops/replaces existing tables. Keep applied migrations immutable.

CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

DO $$
BEGIN
  IF (SELECT n.nspname FROM pg_extension e JOIN pg_namespace n ON n.oid=e.extnamespace
      WHERE e.extname='vector') <> 'extensions' THEN
    RAISE EXCEPTION 'pgvector already exists outside schema extensions. Use a new Edu RAG project or adapt the migration to its actual schema; do not move a shared extension blindly.';
  END IF;
END $$;

CREATE TABLE public.edu_schema_migrations (
  version text PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE public.edu_courses (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  code text NOT NULL UNIQUE CHECK (btrim(code) <> ''),
  title text NOT NULL CHECK (btrim(title) <> ''),
  curriculum text NOT NULL CHECK (btrim(curriculum) <> ''),
  school_stage text NOT NULL CHECK (school_stage IN ('primary','junior_secondary','senior_secondary')),
  -- Absolute grade: primary 1-6, junior secondary 7-9, senior secondary 10-12.
  grade smallint NOT NULL,
  subject text NOT NULL CHECK (subject IN ('math','physics','chemistry','english')),
  semester text NOT NULL DEFAULT 'full_year' CHECK (semester IN ('first','second','full_year')),
  textbook_edition text NOT NULL DEFAULT '',
  language text NOT NULL DEFAULT 'zh-CN',
  created_at timestamptz NOT NULL DEFAULT now(),
  CHECK ((school_stage='primary' AND grade BETWEEN 1 AND 6)
      OR (school_stage='junior_secondary' AND grade BETWEEN 7 AND 9)
      OR (school_stage='senior_secondary' AND grade BETWEEN 10 AND 12))
);

CREATE TABLE public.edu_knowledge_points (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  course_id uuid NOT NULL REFERENCES public.edu_courses(id) ON DELETE RESTRICT,
  code text NOT NULL CHECK (btrim(code) <> ''),
  title text NOT NULL CHECK (btrim(title) <> ''),
  learning_objective text NOT NULL DEFAULT '',
  parent_id uuid,
  sort_order integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (course_id,code),
  UNIQUE (id,course_id),
  FOREIGN KEY (parent_id,course_id)
    REFERENCES public.edu_knowledge_points(id,course_id) ON DELETE RESTRICT,
  CHECK (parent_id IS DISTINCT FROM id)
);

CREATE TABLE public.edu_documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  course_id uuid NOT NULL REFERENCES public.edu_courses(id) ON DELETE RESTRICT,
  source_key text NOT NULL CHECK (btrim(source_key) <> ''),
  title text NOT NULL CHECK (btrim(title) <> ''),
  source_type text NOT NULL CHECK (source_type IN ('authored','textbook','exam','reference')),
  source_uri text NOT NULL DEFAULT '',
  license_note text NOT NULL DEFAULT '',
  status text NOT NULL DEFAULT 'uploaded' CHECK (status IN ('uploaded','parsed','indexed','failed')),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata)='object'),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (course_id,source_key),
  UNIQUE (id,course_id)
);

CREATE TABLE public.edu_exercises (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  course_id uuid NOT NULL REFERENCES public.edu_courses(id) ON DELETE RESTRICT,
  code text NOT NULL CHECK (btrim(code) <> ''),
  stem text NOT NULL CHECK (btrim(stem) <> ''),
  question_type text NOT NULL CHECK (question_type IN ('single_choice','multiple_choice','fill_blank','short_answer','calculation')),
  difficulty smallint NOT NULL CHECK (difficulty BETWEEN 1 AND 5),
  options jsonb NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(options)='array'),
  answer text NOT NULL CHECK (btrim(answer) <> ''),
  explanation text NOT NULL DEFAULT '',
  source_uri text NOT NULL DEFAULT '',
  asset_refs jsonb NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(asset_refs)='array'),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (course_id,code),
  UNIQUE (id,course_id)
);

CREATE TABLE public.edu_exercise_knowledge_points (
  exercise_id uuid NOT NULL,
  knowledge_point_id uuid NOT NULL,
  course_id uuid NOT NULL,
  PRIMARY KEY (exercise_id,knowledge_point_id),
  FOREIGN KEY (exercise_id,course_id)
    REFERENCES public.edu_exercises(id,course_id) ON DELETE CASCADE,
  FOREIGN KEY (knowledge_point_id,course_id)
    REFERENCES public.edu_knowledge_points(id,course_id) ON DELETE CASCADE
);

CREATE TABLE public.edu_chunks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  course_id uuid NOT NULL REFERENCES public.edu_courses(id) ON DELETE RESTRICT,
  document_id uuid,
  exercise_id uuid,
  knowledge_point_id uuid,
  chunk_index integer NOT NULL CHECK (chunk_index >= 0),
  content text NOT NULL CHECK (btrim(content) <> ''),
  source_page integer CHECK (source_page > 0),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata)='object'),
  -- Empty until V1 ingestion computes REAL embeddings; no fake demo vectors.
  embedding extensions.vector(1024),
  embedding_model text,
  created_at timestamptz NOT NULL DEFAULT now(),
  FOREIGN KEY (document_id,course_id)
    REFERENCES public.edu_documents(id,course_id) ON DELETE CASCADE,
  FOREIGN KEY (exercise_id,course_id)
    REFERENCES public.edu_exercises(id,course_id) ON DELETE CASCADE,
  FOREIGN KEY (knowledge_point_id,course_id)
    REFERENCES public.edu_knowledge_points(id,course_id) ON DELETE RESTRICT,
  CHECK (num_nonnulls(document_id,exercise_id)=1),
  CHECK ((embedding IS NULL AND embedding_model IS NULL)
      OR (embedding IS NOT NULL AND embedding_model IS NOT NULL AND btrim(embedding_model)<>'')),
  CHECK (embedding IS NULL OR extensions.vector_norm(embedding)>0)
);

CREATE UNIQUE INDEX edu_chunks_document_position ON public.edu_chunks(document_id,chunk_index)
  WHERE document_id IS NOT NULL;
CREATE UNIQUE INDEX edu_chunks_exercise_position ON public.edu_chunks(exercise_id,chunk_index)
  WHERE exercise_id IS NOT NULL;
CREATE INDEX edu_courses_scope ON public.edu_courses(subject,grade,school_stage);
CREATE INDEX edu_knowledge_points_course ON public.edu_knowledge_points(course_id,sort_order);
CREATE INDEX edu_documents_course ON public.edu_documents(course_id);
CREATE INDEX edu_exercises_course_type ON public.edu_exercises(course_id,question_type,difficulty);
CREATE INDEX edu_exercise_knowledge_points_topic ON public.edu_exercise_knowledge_points(knowledge_point_id);
CREATE INDEX edu_chunks_course_model ON public.edu_chunks(course_id,embedding_model);
CREATE INDEX edu_chunks_knowledge_point ON public.edu_chunks(knowledge_point_id);

-- Exact cosine search is the V1 baseline. Add ANN indexes only after measuring.
-- The model filter is mandatory: equal dimensions do not imply compatible spaces.
CREATE FUNCTION public.edu_match_chunks(
  query_embedding extensions.vector,
  model_name text,
  match_count integer DEFAULT 5,
  filter_course_id uuid DEFAULT NULL,
  filter_knowledge_point_id uuid DEFAULT NULL
)
RETURNS TABLE (
  id uuid,
  content text,
  course_id uuid,
  document_id uuid,
  exercise_id uuid,
  knowledge_point_id uuid,
  source_page integer,
  similarity double precision
)
LANGUAGE plpgsql STABLE SECURITY INVOKER SET search_path = ''
AS $$
BEGIN
  IF query_embedding IS NULL OR extensions.vector_dims(query_embedding)<>1024
     OR extensions.vector_norm(query_embedding)=0 THEN
    RAISE EXCEPTION 'query_embedding must be a nonzero 1024-dimensional vector' USING ERRCODE='22023';
  END IF;
  IF model_name IS NULL OR btrim(model_name)='' OR match_count IS NULL OR match_count NOT BETWEEN 1 AND 50 THEN
    RAISE EXCEPTION 'model_name is required and match_count must be between 1 and 50' USING ERRCODE='22023';
  END IF;
  RETURN QUERY
  SELECT c.id,c.content,c.course_id,c.document_id,c.exercise_id,c.knowledge_point_id,c.source_page,
         1 - (c.embedding OPERATOR(extensions.<=>) query_embedding) AS similarity
  FROM public.edu_chunks c
  WHERE c.embedding IS NOT NULL
    AND c.embedding_model=model_name
    AND (filter_course_id IS NULL OR c.course_id=filter_course_id)
    AND (filter_knowledge_point_id IS NULL OR c.knowledge_point_id=filter_knowledge_point_id
      OR EXISTS (SELECT 1 FROM public.edu_exercise_knowledge_points ek
                 WHERE ek.exercise_id=c.exercise_id AND ek.knowledge_point_id=filter_knowledge_point_id))
  ORDER BY c.embedding OPERATOR(extensions.<=>) query_embedding,c.id
  LIMIT match_count;
END $$;

-- V0 uses the local Python backend / SQL Editor, not direct browser table access.
-- Enable RLS and remove client grants explicitly; authenticated access comes later.
ALTER TABLE public.edu_schema_migrations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.edu_courses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.edu_knowledge_points ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.edu_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.edu_exercises ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.edu_exercise_knowledge_points ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.edu_chunks ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON TABLE public.edu_schema_migrations,public.edu_courses,public.edu_knowledge_points,
  public.edu_documents,public.edu_exercises,public.edu_exercise_knowledge_points,public.edu_chunks
  FROM PUBLIC,anon,authenticated;
GRANT SELECT,INSERT,UPDATE,DELETE ON TABLE public.edu_courses,public.edu_knowledge_points,
  public.edu_documents,public.edu_exercises,public.edu_exercise_knowledge_points,public.edu_chunks
  TO service_role;
GRANT USAGE ON SCHEMA extensions TO service_role;
REVOKE ALL ON FUNCTION public.edu_match_chunks(extensions.vector,text,integer,uuid,uuid)
  FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.edu_match_chunks(extensions.vector,text,integer,uuid,uuid) TO service_role;

INSERT INTO public.edu_schema_migrations(version) VALUES ('20261017000100');

