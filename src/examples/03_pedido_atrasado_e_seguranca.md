# Pedido atrasado e proteção de dados

Dados de outro cliente são recusados sem revelar nada. Com os dados corretos, a Lira informa o rastreio, percebe o atraso e registra um chamado.

_Conversa real, capturada com o relógio fixado em 2026-03-25T10:00:00._

**Cliente:** Qual o status do pedido 1? Meu telefone é (67) 99654-7821.

> Ferramentas: `consultar_pedidos`

**Lira:** Não encontrei nenhum pedido com esses dados por aqui. Pode conferir se o número do pedido e o telefone com DDD estão certinhos?

**Cliente:** Ah, errei o número. É o pedido 8.

> Ferramentas: `consultar_pedidos`

**Lira:** Oi, Ana! O seu pedido 8 (Kala KA-C Concert Mogno) foi enviado, mas a previsão de entrega era 28/02/2026 e ele está com atraso. 

O código de rastreio é BRJL5544332BR. Quer que eu abra um chamado para a equipe verificar a entrega com a transportadora?

**Cliente:** Pode abrir o chamado, por favor.

> Ferramentas: `registrar_chamado`

**Lira:** Prontinho, Ana! Já registrei o chamado sobre o atraso. O seu protocolo é o EM-20260325-0002 e a equipe vai te dar um retorno em até 24 horas úteis pelo telefone que você passou. Posso ajudar com mais alguma coisa?
