# Revision y respuestas al plan de saneamiento ALTTEZ

Fecha: 2026-08-03
Documento revisado: `Plan de saneamiento - ALTTEZ`, basado en la auditoria tecnica
del mismo dia.

## Dictamen

El plan es una buena traduccion de la auditoria a fases ejecutables. Se aprueba
como base, pero no debe ejecutarse literalmente sin cinco ajustes:

1. El hash de un artefacto comprueba integridad, no autenticidad.
2. Pydantic real debe adelantarse a fase 0/1 temprana, no quedar detras de tareas
   cosmeticas.
3. Las lambdas del comite tienen uso operativo y M9 no es solo documentacion.
4. Crear un `pyproject.toml` minimo es habilitador de la limpieza, no una tarea que
   deba evitarse.
5. El Mundial 2026 ya termino, pero su funcionalidad esta integrada en el producto;
   debe retirarse o transformarse con una decision explicita, no archivarse por
   fecha automaticamente.

## Respuestas a los vacios

### 1. Origen del Pydantic local

Estado: parcialmente resuelto.

Git muestra que `pydantic/__init__.py` fue introducido completo en el commit
`9a58eb4` del 2026-05-26, junto con `ingestion/contracts.py` y la certificacion de
fase 2. No hubo una evolucion gradual ni existe comentario de decision. Pydantic no
aparece en `requirements.txt` antes ni despues del commit.

Inferencia razonable: se creo deliberadamente para habilitar contratos sin agregar
la dependencia. No hay evidencia para afirmar si fue por compatibilidad, peso o
urgencia.

Decision propuesta: no bloquear el saneamiento esperando una motivacion que Git no
contiene. Preservar la semantica observable con pruebas de caracterizacion y migrar.

### 2. Esfuerzo y riesgo

| Accion | Esfuerzo | Riesgo de regresion | Prioridad |
| --- | --- | --- | --- |
| Corregir 5 errores Ruff | Bajo | Bajo | Inmediata |
| Hacer Ruff bloqueante | Bajo | Bajo tras corregir | Inmediata |
| Verificar tamaño/hash | Bajo | Medio por artefactos legacy | Inmediata |
| Restringir/autenticar artefactos | Medio | Medio | Alta |
| Separar tests lentos | Bajo-medio | Bajo | Alta |
| Caracterizar Pydantic local | Medio | Bajo | Alta |
| Migrar a Pydantic v2 | Medio | Medio-alto | Alta |
| Crear `pyproject.toml` minimo | Medio | Medio | Alta |
| Tipar estados de disponibilidad | Medio | Medio | Alta |
| Dividir monolitos | Alto | Alto | Gradual |
| Archivar legacy | Bajo-medio | Medio si hay usuarios externos | Tras confirmar |

### 3. Ventana Mundial 2026

Confirmado: la final se disputo el 19 de julio de 2026; a fecha del informe el
torneo termino.

Correccion al plan: baja la urgencia de nuevas features especificas del Mundial,
pero no convierte el modulo en archivo muerto. Hay referencias activas en:

- dashboard y selector principal;
- monitoreo;
- router/contexto Klement;
- scripts de simulacion y brief;
- documentacion y configuracion.

Decision propuesta:

- congelar nuevas features del Mundial;
- conservar durante una ventana de consulta post-torneo;
- decidir si el simulador pasa a modo historico/replay o se retira de la UI;
- mantener componentes genericos reutilizables para torneos de 48 equipos;
- no mezclar esta retirada con el refactor del router.

### 4. Rollback del saneamiento

Se acepta el vacio. Estrategia propuesta:

- cambios pequeños y commits independientes;
- feature flags solo donde haya dos comportamientos productivos reales;
- artefactos: aceptar temporalmente filas sin hash con alerta, backfill de hash,
  luego activar rechazo estricto;
- Pydantic: mover el shim a `alttez.compat.pydantic_legacy`, ejecutar fixtures
  contra ambos validadores y cortar solo cuando las diferencias esten clasificadas;
- refactors: fachada estable y pruebas de caracterizacion antes de mover logica;
- rollback operativo por version/commit, no dos implementaciones permanentes.

### 5. Licencias de datos

Estado: riesgo confirmado, permiso comercial no demostrado.

- StatsBomb Open Data indica uso libre orientado a investigacion/interes genuino y
  exige atribucion visible al publicar o distribuir analisis. Eso no equivale a una
  licencia comercial general confirmada.
- football-data.org pide un disclaimer visible de atribucion.
- API-Football permite construir aplicaciones, pero prohibe reventa directa de sus
  datos y advierte que betting, medios, logos y datos de competiciones pueden
  requerir licencias de titulares.

En ALTTEZ, API-Football esta marcado como retirado por clave muerta, aunque el
provider permanece. Su revision sigue siendo necesaria antes de reactivarlo.

Decision propuesta: crear un registro de fuentes con proveedor, dato consumido,
atribucion, plan contratado, retencion, redistribucion y uso comercial permitido.
La conclusion legal final requiere revision humana/juridica.

### 6. Gates

Se acepta, con ajustes:

- sustituir “dos semanas” por siete corridas diarias consecutivas verdes y al
  menos un despliegue manual de prueba;
- definir ahora el tiempo objetivo unitario: menos de 30 segundos sin PyMC, red ni
  base real;
- no exigir coverage perfecto: medir baseline y evitar regresion;
- canary solo es valido si existe realmente un segundo entorno o audiencia.

### 7. Owners

Git identifica a Julian como autor principal de los cambios auditados, pero eso no
demuestra disponibilidad ni ownership actual.

Decision pendiente humana:

| Area | Owner requerido |
| --- | --- |
| Contratos/ingestion | Por asignar |
| Inferencia/modelos | Por asignar |
| Datos/Supabase | Por asignar |
| Dashboard/reportes | Por asignar |
| Plataforma/CI | Por asignar |
| Producto/licencias | Por asignar |

Una persona puede ocupar varios roles, pero cada fase necesita un aprobador.

### 8. Freeze de features

Se acepta solo de forma focalizada. Un freeze global retrasaria correcciones
operativas.

Propuesta:

- freeze de nuevas features en `live_matches.py`, `router.py`,
  `prediction_memory.py` y dashboard mientras se extrae cada limite;
- excepciones para incidentes, seguridad y datos incorrectos;
- duracion por modulo, no por toda la fase 2;
- rama corta e integracion frecuente.

## Respuestas a preguntas abiertas

### Contratos e ingestion

**Por que existe el shim:** no documentado. Fue creado en un solo commit junto con
los contratos, sin declarar Pydantic real.

**Diferencias conocidas:** el shim implementa una fraccion de Pydantic. Acepta
algunos `None` no opcionales, tiene serializacion anidada parcial y cubre una parte
limitada de coercion/configuracion. Las diferencias exactas deben convertirse en
una matriz de tests.

**Consumidores externos:** no hay evidencia dentro del repositorio. Solo
`ingestion/contracts.py` importa Pydantic. La existencia de otros repositorios o
servicios requiere confirmacion humana.

### Modelos y artefactos

**Permisos:** las migraciones no contienen `GRANT`, `REVOKE`, RLS ni policies para
`model_artifacts`. Los permisos efectivos viven en Supabase y no pueden inferirse
del repo. Debe auditarse el rol asociado a `SUPABASE_DB_URL` y los secrets de CI.

**Vector real:** el pipeline diario tiene credenciales capaces de subir artefactos.
El riesgo no es puramente teorico, pero no hay evidencia de compromiso. El modelo
de amenazas depende de quien pueda modificar secrets, runners o la base.

**Lambdas:** son operativas. Alimentan xG de dashboard, persistencia, Over 2.5,
total esperado y total mas probable en brief. M9 sube a prioridad media-alta.

### Arquitectura

**Modulo mas caliente en 90 dias:** `training/challenger_cycle.py` con 12 commits.
Router, memoria, dashboard y live matches tienen 9 cada uno. La secuencia no debe
basarse solo en calor: primero memoria/errores por observabilidad, luego challenger
por frecuencia de cambio.

**Consumidores externos del dashboard:** no identificables desde el repo.

**Incidentes ocultos como “sin datos”:** no hay registro local suficiente para
probar incidentes historicos. El patron de codigo permite que ocurran; por eso debe
corregirse sin esperar un incidente confirmado.

### CI y dependencias

**Ruff no bloqueante:** el comentario de CI dice expresamente “Non-blocking for
now”. Fue introducido el 2026-06-16. Es una medida temporal que quedo vigente, no
una justificacion por falsos positivos.

**Dependencias externas:** no pueden descartarse para notebooks/repositorios que no
estan presentes. Dentro del repo varias tienen cero referencias.

**Entorno CI:** confirmado por YAML: Ubuntu y Python 3.13. El estado del ultimo run
remoto no quedo confirmado en esta revision.

### Codigo legacy y residuos

**Uso del laboratorio:** no hay consumidores productivos encontrados para scraper,
logistic ni el wrapper XGBoost antiguo. `trainer/tuner` conservan pruebas y deben
decidirse como conjunto.

**`archived_files/`:** contiene una carpeta y un `driver_fixing.lock` de cero bytes;
no esta versionado. Es candidato seguro a eliminacion tras confirmar que ningun
proceso externo lo usa.

**`tmptvhsjedb/`:** Git lo ignora, no esta versionado y el sistema nego acceso a su
contenido. No se declara seguro eliminarlo sin inspeccion con permisos adecuados.

### Producto y negocio

**Licencias:** requieren cierre juridico antes de uso comercial o de apuestas.

**SLA:** no hay SLA, owner ni compromiso de disponibilidad documentado. Debe
responderlo producto/operacion antes de definir ventanas de cambio.

## Correcciones a la matriz

### Hacer ya

- corregir Ruff y hacerlo bloqueante;
- medir baseline de tests y coverage sin fijar aun un umbral;
- separar la prueba PyMC y dependencias externas del unit suite;
- caracterizar contratos del Pydantic local;
- verificar tamaño/hash y diseñar autenticidad/permisos de artefactos;
- cambiar errores silenciosos prioritarios por estados observables;
- congelar nuevas features especificas del Mundial.

### Fase siguiente

- migrar a Pydantic real;
- crear `pyproject.toml` minimo y extras;
- restringir permisos de artefactos;
- corregir semantica de lambdas;
- clasificar/eliminar legacy;
- normalizar documentacion con mojibake.

### Gradual

- repositorios tipados;
- dividir `live_matches`, dashboard, router y challenger;
- templates HTML/CSS;
- type checking y cobertura incremental.

### No hacer

- reescritura completa;
- freeze global indefinido;
- coverage/type checking al 100% como primer gate;
- shadow mode permanente;
- eliminar el modulo Mundial solo porque termino el torneo;
- confiar en SHA-256 almacenado junto al blob como control de autenticidad.

## Fases ajustadas

### Fase 0A: baseline y CI

Gate:

- cero Ruff;
- unit suite menor a 30 segundos;
- PyMC/red/DB fuera del unit suite;
- siete corridas diarias verdes;
- baseline de coverage publicado.

### Fase 0B: seguridad y observabilidad

Gate:

- artefactos con tamaño/hash verificados y politica legacy definida;
- permisos efectivos de Supabase documentados;
- errores de registro/modelos distinguibles de “sin datos”;
- alerta segura sin secretos.

### Fase 1: contratos y dependencias

Gate:

- matriz de diferencias legacy/Pydantic v2 cerrada;
- contratos v2 activos;
- `pyproject.toml` instala el proyecto desde cero;
- lock reproducible y extras funcionales;
- rollback ensayado.

### Fase 2: higiene y producto

Gate:

- owner de cada modulo declarado;
- decision sobre experiencia post-Mundial;
- codigo legacy clasificado como soportado, laboratorio o eliminado;
- registro de licencias aprobado.

### Fase 3: limites arquitectonicos

Gate:

- fachadas compatibles;
- pruebas de caracterizacion;
- canary si existe entorno real;
- sin regresion de metricas operativas.

## Decisiones humanas necesarias

1. Uso interno, comercial, apuestas o combinacion.
2. Owner y aprobador de cada area.
3. SLA del dashboard y brief.
4. Existencia de consumidores externos.
5. Acceso efectivo a Supabase y GitHub secrets.
6. Retener el simulador Mundial como historico, generalizarlo o retirarlo.
7. Conservacion del laboratorio legacy.

Sin estas siete respuestas se puede ejecutar Fase 0A, pero no cerrar seguridad,
licencias, eliminacion de legacy ni producto post-Mundial.
