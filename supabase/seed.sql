-- Authored examples only. These records are NOT an official curriculum.
-- Re-running preserves any edits you made to existing demo records.

INSERT INTO public.edu_courses(id,code,title,curriculum,school_stage,grade,subject,textbook_edition)
VALUES ('10000000-0000-4000-8000-000000000001','DEMO-CN-G3-MATH','三年级数学演示课程',
        '自编演示课程（非官方课标）','primary',3,'math','自编演示资料')
ON CONFLICT DO NOTHING;

INSERT INTO public.edu_knowledge_points(id,course_id,code,title,learning_objective,sort_order)
VALUES
 ('20000000-0000-4000-8000-000000000001','10000000-0000-4000-8000-000000000001',
  'rectangle-area','长方形的面积','能使用长乘宽计算长方形的面积，并写出平方单位。',1),
 ('20000000-0000-4000-8000-000000000002','10000000-0000-4000-8000-000000000001',
  'unit-fraction','几分之一','能解释将整体平均分成若干份后，一份所表示的分数。',2)
ON CONFLICT DO NOTHING;

INSERT INTO public.edu_documents(id,course_id,source_key,title,source_type,source_uri,license_note,status)
VALUES ('30000000-0000-4000-8000-000000000001','10000000-0000-4000-8000-000000000001',
        'demo-math-g3-v0','数学演示知识卡片','authored','data/samples/math_g3.md',
        '本项目原创演示内容，可用于学习和测试。','parsed')
ON CONFLICT DO NOTHING;

INSERT INTO public.edu_exercises(id,course_id,code,stem,question_type,difficulty,answer,explanation)
VALUES
 ('40000000-0000-4000-8000-000000000001','10000000-0000-4000-8000-000000000001',
  'DEMO-AREA-001','一个长方形长 8 厘米、宽 3 厘米，它的面积是多少？','calculation',1,
  '24 平方厘米','长方形的面积 = 长 × 宽 = 8 × 3 = 24，面积单位是平方厘米。'),
 ('40000000-0000-4000-8000-000000000002','10000000-0000-4000-8000-000000000001',
  'DEMO-AREA-002','一个长方形面积为 35 平方米，宽为 5 米，长是多少？','calculation',2,
  '7 米','长 = 面积 ÷ 宽 = 35 ÷ 5 = 7，长度单位是米。'),
 ('40000000-0000-4000-8000-000000000003','10000000-0000-4000-8000-000000000001',
  'DEMO-FRACTION-001','把一张纸平均分成 4 份，每份是这张纸的几分之几？','fill_blank',1,
  '四分之一（1/4）','分母 4 表示平均分成 4 份，分子 1 表示其中的 1 份。')
ON CONFLICT DO NOTHING;

INSERT INTO public.edu_exercise_knowledge_points(exercise_id,knowledge_point_id,course_id)
VALUES
 ('40000000-0000-4000-8000-000000000001','20000000-0000-4000-8000-000000000001','10000000-0000-4000-8000-000000000001'),
 ('40000000-0000-4000-8000-000000000002','20000000-0000-4000-8000-000000000001','10000000-0000-4000-8000-000000000001'),
 ('40000000-0000-4000-8000-000000000003','20000000-0000-4000-8000-000000000002','10000000-0000-4000-8000-000000000001')
ON CONFLICT DO NOTHING;

INSERT INTO public.edu_chunks(id,course_id,document_id,knowledge_point_id,chunk_index,content,metadata)
VALUES
 ('50000000-0000-4000-8000-000000000001','10000000-0000-4000-8000-000000000001',
  '30000000-0000-4000-8000-000000000001','20000000-0000-4000-8000-000000000001',0,
  '长方形的面积等于长乘宽。计算前，长和宽应使用相同的长度单位。长和宽以厘米为单位时，面积以平方厘米为单位。例如长 8 厘米、宽 3 厘米，面积是 24 平方厘米。',
  '{"heading":"长方形的面积","sample":true}'::jsonb),
 ('50000000-0000-4000-8000-000000000002','10000000-0000-4000-8000-000000000001',
  '30000000-0000-4000-8000-000000000001','20000000-0000-4000-8000-000000000002',1,
  '把一个整体平均分成若干份，其中一份可以用几分之一表示。分母表示平均分成的份数。例如把一张纸平均分成 4 份，每份是这张纸的四分之一，写作 1/4。',
  '{"heading":"几分之一","sample":true}'::jsonb)
ON CONFLICT DO NOTHING;

-- Exercise retrieval text contains the stem; answer and explanation stay in the
-- exercise record, so a later practice UI can reveal them separately.
INSERT INTO public.edu_chunks(id,course_id,exercise_id,knowledge_point_id,chunk_index,content,metadata)
SELECT
  CASE e.code
    WHEN 'DEMO-AREA-001' THEN '50000000-0000-4000-8000-000000000003'::uuid
    WHEN 'DEMO-AREA-002' THEN '50000000-0000-4000-8000-000000000004'::uuid
    ELSE '50000000-0000-4000-8000-000000000005'::uuid
  END,
  e.course_id,e.id,ek.knowledge_point_id,0,e.stem,'{"sample":true}'::jsonb
FROM public.edu_exercises e
JOIN public.edu_exercise_knowledge_points ek ON ek.exercise_id=e.id
WHERE e.id IN ('40000000-0000-4000-8000-000000000001',
               '40000000-0000-4000-8000-000000000002',
               '40000000-0000-4000-8000-000000000003')
ON CONFLICT DO NOTHING;

