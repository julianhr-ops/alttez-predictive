# Mantenimiento del contexto para IA

## Checklist por cambio

| Cambio | Documentos a revisar |
| --- | --- |
| Nuevo modulo o limite | `architecture.md`, `README.md` |
| Nuevo comando o CI | `workflows.md`, `AGENTS.md` |
| Nueva variable | `.env.example`, `configuration.md`, workflow |
| Feature, modelo o payload | `data-and-models.md`, `architecture.md` |
| Nueva regla obligatoria | `AGENTS.md` |

## Regla de actualizacion

Documentar invariantes y decisiones estables, no cada funcion. Antes de afirmar que
una herramienta, migracion, rama o provider existe, verificar:

```powershell
git branch --show-current
Get-ChildItem sql/supabase/migrations
Get-Content .github/workflows/ci.yml
rg "os\.getenv|environ\.get" alttez scripts
```

Evitar datos que envejecen sin explicar como verificarlos.

## Auditoria periodica

1. Comparar `requirements.txt`, CI y comandos documentados.
2. Comparar `.env.example` con variables leidas por codigo y workflows.
3. Revisar que train e inference compartan features.
4. Revisar orden de migraciones y pruebas del runner.
5. Buscar enlaces rotos y rutas eliminadas.
6. Confirmar que no haya secretos o datos privados.
7. Registrar fecha y resultado abajo.

## Estado

- Ultima auditoria: 2026-08-03.
- Resultado: se creo `docs/ai/` como fuente neutral. El anterior `CLAUDE.md`
  presentaba codificacion danada y datos obsoletos sobre Ruff, migraciones y rama.
- Codigo fuente modificado por esta auditoria: ninguno.
