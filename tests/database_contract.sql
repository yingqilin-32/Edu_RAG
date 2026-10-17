-- Integration assertions. Run only against a disposable test database with seed.sql.
-- All vector fixtures are synthetic test data and are rolled back.
BEGIN;

DO $$
BEGIN
  IF (SELECT count(*) FROM public.edu_chunks WHERE embedding IS NOT NULL)<>0 THEN
    RAISE EXCEPTION 'demo data must not contain fake embeddings';
  END IF;
  IF (SELECT count(*) FROM public.edu_exercises)<>3 THEN
    RAISE EXCEPTION 'seed exercise count must equal 3';
  END IF;
END $$;

-- Same-dimension vectors from different models must never enter one result set.
UPDATE public.edu_chunks
SET embedding=('[' || '1,' || repeat('0,',1022) || '0]')::extensions.vector,
    embedding_model='contract-model-a'
WHERE id='50000000-0000-4000-8000-000000000001';
UPDATE public.edu_chunks
SET embedding=('[' || '1,' || repeat('0,',1022) || '0]')::extensions.vector,
    embedding_model='contract-model-b'
WHERE id='50000000-0000-4000-8000-000000000002';

DO $$
DECLARE
  test_vector extensions.vector := ('[' || '1,' || repeat('0,',1022) || '0]')::extensions.vector;
  result_id uuid;
  result_count integer;
BEGIN
  SELECT count(*),min(id::text)::uuid INTO result_count,result_id
  FROM public.edu_match_chunks(test_vector,'contract-model-a');
  IF result_count<>1 OR result_id<>'50000000-0000-4000-8000-000000000001' THEN
    RAISE EXCEPTION 'model isolation failed';
  END IF;
  SELECT count(*) INTO result_count FROM public.edu_match_chunks(
    test_vector,'contract-model-a',5,'10000000-0000-4000-8000-000000000099');
  IF result_count<>0 THEN RAISE EXCEPTION 'course filter failed'; END IF;
  SELECT count(*) INTO result_count FROM public.edu_match_chunks(
    test_vector,'contract-model-a',5,NULL,'20000000-0000-4000-8000-000000000002');
  IF result_count<>0 THEN RAISE EXCEPTION 'knowledge point filter failed'; END IF;
  IF (SELECT similarity FROM public.edu_match_chunks(test_vector,'contract-model-a'))<0.999 THEN
    RAISE EXCEPTION 'cosine similarity failed';
  END IF;
  BEGIN
    PERFORM public.edu_match_chunks('[1,0]'::extensions.vector,'contract-model-a');
    RAISE EXCEPTION 'wrong-dimensional query was accepted';
  EXCEPTION WHEN invalid_parameter_value THEN NULL;
  END;
  BEGIN
    PERFORM public.edu_match_chunks(test_vector,'contract-model-a',0);
    RAISE EXCEPTION 'invalid match_count was accepted';
  EXCEPTION WHEN invalid_parameter_value THEN NULL;
  END;
  BEGIN
    UPDATE public.edu_chunks SET exercise_id='40000000-0000-4000-8000-000000000001'
    WHERE id='50000000-0000-4000-8000-000000000001';
    RAISE EXCEPTION 'chunk with two source owners was accepted';
  EXCEPTION WHEN check_violation THEN NULL;
  END;
  BEGIN
    UPDATE public.edu_chunks SET embedding_model=NULL
    WHERE id='50000000-0000-4000-8000-000000000001';
    RAISE EXCEPTION 'embedding without model name was accepted';
  EXCEPTION WHEN check_violation THEN NULL;
  END;
  BEGIN
    UPDATE public.edu_exercises SET difficulty=6
    WHERE id='40000000-0000-4000-8000-000000000001';
    RAISE EXCEPTION 'invalid difficulty was accepted';
  EXCEPTION WHEN check_violation THEN NULL;
  END;
END $$;

INSERT INTO public.edu_courses(id,code,title,curriculum,school_stage,grade,subject)
VALUES ('10000000-0000-4000-8000-000000000099','CONTRACT-G9','测试课程','测试','junior_secondary',9,'math');
DO $$
BEGIN
  BEGIN
    UPDATE public.edu_chunks SET course_id='10000000-0000-4000-8000-000000000099'
    WHERE id='50000000-0000-4000-8000-000000000001';
    RAISE EXCEPTION 'cross-course source reference was accepted';
  EXCEPTION WHEN foreign_key_violation THEN NULL;
  END;
END $$;

-- Grants, RLS, and function privileges must all block direct browser roles.
DO $$
DECLARE table_name text; role_name text;
BEGIN
  FOREACH table_name IN ARRAY ARRAY['edu_courses','edu_knowledge_points','edu_documents',
    'edu_exercises','edu_exercise_knowledge_points','edu_chunks','edu_schema_migrations']
  LOOP
    IF NOT (SELECT relrowsecurity FROM pg_class WHERE oid=('public.'||table_name)::regclass) THEN
      RAISE EXCEPTION 'RLS missing for table %',table_name;
    END IF;
    FOREACH role_name IN ARRAY ARRAY['anon','authenticated'] LOOP
      IF has_table_privilege(role_name,'public.'||table_name,'SELECT,INSERT,UPDATE,DELETE') THEN
        RAISE EXCEPTION 'unexpected browser table privilege for %',role_name;
      END IF;
    END LOOP;
  END LOOP;
  FOREACH role_name IN ARRAY ARRAY['anon','authenticated'] LOOP
    IF has_function_privilege(role_name,'public.edu_match_chunks(extensions.vector,text,integer,uuid,uuid)','EXECUTE') THEN
      RAISE EXCEPTION 'unexpected browser function privilege for %',role_name;
    END IF;
  END LOOP;
END $$;

ROLLBACK;

