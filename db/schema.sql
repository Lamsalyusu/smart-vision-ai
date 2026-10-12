--
-- PostgreSQL database dump
--

\restrict n56rjS9fAs4mchdevCpyEjXqGHZZFX89YJX6t9LlX3ilwUGsx1eHoDgQshoz9OF

-- Dumped from database version 17.6
-- Dumped by pg_dump version 18.6

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA public;


--
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON SCHEMA public IS 'standard public schema';


--
-- Name: update_updated_at_column(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.update_updated_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: detection_crops; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.detection_crops (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid,
    object_class text NOT NULL,
    confidence double precision,
    crop_image text NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: detection_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.detection_history (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid,
    object_class text NOT NULL,
    confidence double precision NOT NULL,
    bbox jsonb,
    crop_image text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: detection_results; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.detection_results (
    id uuid DEFAULT extensions.uuid_generate_v4() NOT NULL,
    session_id uuid NOT NULL,
    user_id uuid NOT NULL,
    image_url text,
    detected_objects jsonb NOT NULL,
    total_objects_detected integer DEFAULT 0,
    processing_time_ms integer,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: detection_sessions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.detection_sessions (
    id uuid DEFAULT extensions.uuid_generate_v4() NOT NULL,
    user_id uuid NOT NULL,
    session_name character varying(255),
    session_type character varying(50) DEFAULT 'image'::character varying,
    started_at timestamp with time zone DEFAULT now(),
    ended_at timestamp with time zone,
    total_detections integer DEFAULT 0,
    is_active boolean DEFAULT true,
    CONSTRAINT detection_sessions_session_type_check CHECK (((session_type)::text = ANY ((ARRAY['image'::character varying, 'realtime'::character varying])::text[])))
);


--
-- Name: llm_queries; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.llm_queries (
    id uuid DEFAULT extensions.uuid_generate_v4() NOT NULL,
    user_id uuid NOT NULL,
    detection_result_id uuid,
    query_text text NOT NULL,
    llm_response text NOT NULL,
    context_objects jsonb,
    processing_time_ms integer,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: models_3d; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.models_3d (
    id uuid DEFAULT extensions.uuid_generate_v4() NOT NULL,
    object_class character varying(100) NOT NULL,
    model_name character varying(255) NOT NULL,
    model_url text NOT NULL,
    thumbnail_url text,
    file_size_mb numeric(10,2),
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: password_resets; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.password_resets (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid,
    token text NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    used boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: user_sessions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.user_sessions (
    id uuid DEFAULT extensions.uuid_generate_v4() NOT NULL,
    user_id uuid NOT NULL,
    session_token character varying(255) NOT NULL,
    ip_address character varying(50),
    user_agent text,
    expires_at timestamp with time zone NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    is_active boolean DEFAULT true
);


--
-- Name: users; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.users (
    id uuid DEFAULT extensions.uuid_generate_v4() NOT NULL,
    full_name character varying(255) NOT NULL,
    email character varying(255) NOT NULL,
    username character varying(100) NOT NULL,
    password_hash character varying(255) NOT NULL,
    is_active boolean DEFAULT true,
    is_verified boolean DEFAULT false,
    profile_image_url text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    last_login timestamp with time zone,
    auth_provider text DEFAULT 'email'::text
);


--
-- Name: detection_crops detection_crops_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.detection_crops
    ADD CONSTRAINT detection_crops_pkey PRIMARY KEY (id);


--
-- Name: detection_history detection_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.detection_history
    ADD CONSTRAINT detection_history_pkey PRIMARY KEY (id);


--
-- Name: detection_results detection_results_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.detection_results
    ADD CONSTRAINT detection_results_pkey PRIMARY KEY (id);


--
-- Name: detection_sessions detection_sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.detection_sessions
    ADD CONSTRAINT detection_sessions_pkey PRIMARY KEY (id);


--
-- Name: llm_queries llm_queries_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.llm_queries
    ADD CONSTRAINT llm_queries_pkey PRIMARY KEY (id);


--
-- Name: models_3d models_3d_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.models_3d
    ADD CONSTRAINT models_3d_pkey PRIMARY KEY (id);


--
-- Name: password_resets password_resets_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.password_resets
    ADD CONSTRAINT password_resets_pkey PRIMARY KEY (id);


--
-- Name: password_resets password_resets_token_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.password_resets
    ADD CONSTRAINT password_resets_token_key UNIQUE (token);


--
-- Name: user_sessions user_sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_sessions
    ADD CONSTRAINT user_sessions_pkey PRIMARY KEY (id);


--
-- Name: user_sessions user_sessions_session_token_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_sessions
    ADD CONSTRAINT user_sessions_session_token_key UNIQUE (session_token);


--
-- Name: users users_email_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_email_key UNIQUE (email);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- Name: idx_detection_results_session_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_detection_results_session_id ON public.detection_results USING btree (session_id);


--
-- Name: idx_detection_results_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_detection_results_user_id ON public.detection_results USING btree (user_id);


--
-- Name: idx_detection_sessions_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_detection_sessions_user_id ON public.detection_sessions USING btree (user_id);


--
-- Name: idx_llm_queries_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_llm_queries_user_id ON public.llm_queries USING btree (user_id);


--
-- Name: idx_models_3d_object_class; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_models_3d_object_class ON public.models_3d USING btree (object_class);


--
-- Name: idx_user_sessions_token; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_user_sessions_token ON public.user_sessions USING btree (session_token);


--
-- Name: idx_user_sessions_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_user_sessions_user_id ON public.user_sessions USING btree (user_id);


--
-- Name: idx_users_email; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_users_email ON public.users USING btree (email);


--
-- Name: idx_users_username; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_users_username ON public.users USING btree (username);


--
-- Name: models_3d update_models_3d_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER update_models_3d_updated_at BEFORE UPDATE ON public.models_3d FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: users update_users_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON public.users FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: detection_crops detection_crops_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.detection_crops
    ADD CONSTRAINT detection_crops_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: detection_history detection_history_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.detection_history
    ADD CONSTRAINT detection_history_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: detection_results detection_results_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.detection_results
    ADD CONSTRAINT detection_results_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.detection_sessions(id) ON DELETE CASCADE;


--
-- Name: detection_results detection_results_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.detection_results
    ADD CONSTRAINT detection_results_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: detection_sessions detection_sessions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.detection_sessions
    ADD CONSTRAINT detection_sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: llm_queries llm_queries_detection_result_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.llm_queries
    ADD CONSTRAINT llm_queries_detection_result_id_fkey FOREIGN KEY (detection_result_id) REFERENCES public.detection_results(id) ON DELETE CASCADE;


--
-- Name: llm_queries llm_queries_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.llm_queries
    ADD CONSTRAINT llm_queries_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: password_resets password_resets_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.password_resets
    ADD CONSTRAINT password_resets_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: user_sessions user_sessions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_sessions
    ADD CONSTRAINT user_sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: models_3d Anyone can view 3D models; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Anyone can view 3D models" ON public.models_3d FOR SELECT USING (true);


--
-- Name: user_sessions Users can create sessions; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can create sessions" ON public.user_sessions FOR INSERT WITH CHECK (true);


--
-- Name: users Users can update own profile; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can update own profile" ON public.users FOR UPDATE USING (true);


--
-- Name: llm_queries Users can view own LLM queries; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can view own LLM queries" ON public.llm_queries USING (true);


--
-- Name: detection_results Users can view own detection results; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can view own detection results" ON public.detection_results USING (true);


--
-- Name: detection_sessions Users can view own detection sessions; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can view own detection sessions" ON public.detection_sessions USING (true);


--
-- Name: users Users can view own profile; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can view own profile" ON public.users FOR SELECT USING (true);


--
-- Name: user_sessions Users can view own sessions; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY "Users can view own sessions" ON public.user_sessions FOR SELECT USING (true);


--
-- Name: detection_results; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.detection_results ENABLE ROW LEVEL SECURITY;

--
-- Name: detection_sessions; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.detection_sessions ENABLE ROW LEVEL SECURITY;

--
-- Name: llm_queries; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.llm_queries ENABLE ROW LEVEL SECURITY;

--
-- Name: models_3d; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.models_3d ENABLE ROW LEVEL SECURITY;

--
-- Name: user_sessions; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.user_sessions ENABLE ROW LEVEL SECURITY;

--
-- Name: users; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

--
-- PostgreSQL database dump complete
--

\unrestrict n56rjS9fAs4mchdevCpyEjXqGHZZFX89YJX6t9LlX3ilwUGsx1eHoDgQshoz9OF

